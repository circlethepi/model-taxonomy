#!/usr/bin/env python
"""Regenerate the whole visualization suite for the simplex3_qwen experiment.

Four taxonomy levels over the 16 Qwen3.5-4B adapters that span a 3-group topic
simplex. Distance matrices in ``copper_r``; MDS embeddings coloured by each
model's own mixture (see :mod:`src.plots.simplex`).

Not every (rung, metric) cell exists, and the gaps are structural rather than
accidental:

* **CKA** needs more than one row, so it cannot run on a ``model mean``
  representation, and :func:`src.notebook.structure.cka_distance_matrix` takes a
  single (layer, projection), so it cannot span a layer grouping.
* **Bures-Wasserstein** stacks per-block factors before its SVD, so every block
  must share an input dim; a selection mixing 2560-input and 4096-input
  projections has no BW value. It is also rank-1, and so carries nothing cosine
  does not, on a single-row representation.

Absent cells are drawn with the reason in place rather than left blank or
silently dropped, so the constraint stays visible in the figure.

Usage
-----
    python scripts/make_simplex3_figures.py                  # everything
    python scripts/make_simplex3_figures.py --level functional
    python scripts/make_simplex3_figures.py --skip-sweep     # omit the slow one
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── BLAS threads, pinned before numpy is imported ────────────────────────────
#   This is a large, measured effect, not a precaution. The Bures-Wasserstein
#   precompute takes a thin SVD of a (n_blocks*r, d) matrix — (384, 2560) for a
#   24-block selection. Left to the default thread count on this 14-core host
#   that SVD measured **79 s** idle and 118 s under load; pinned to one thread it
#   measures **5.3 s**, with results identical to floating-point tolerance. The
#   matrices are small enough that threading buys nothing and the synchronisation
#   dominates. Unpinned, the behavioral level did not finish inside 50 minutes.
#
#   Must be set before numpy loads its BLAS, hence the placement above the import
#   rather than in main(). Override with MODEL_TAXONOMY_THREADS if a host differs.
_THREADS = os.environ.get("MODEL_TAXONOMY_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis import scan_cache  # noqa: E402
from src.analysis.bridge import as_distance_matrix  # noqa: E402
from src.analysis.comparison import _compute_distance_matrix  # noqa: E402
from src.plots.config import set_style  # noqa: E402
from src.plots.figures import save_figure  # noqa: E402
from src.plots.simplex import (  # noqa: E402
    dm_grid, mds_grid, mixture_label, mixture_weights, sort_by_mixture,
    ternary_legend,
)

# ── Experiment coordinates ────────────────────────────────────────────────────
CACHE_ROOT = REPO_ROOT.parent.parent.parent / "results" / "shared_cache"
if not CACHE_ROOT.exists():                      # running from the main checkout
    CACHE_ROOT = REPO_ROOT / "results" / "shared_cache"
ADAPTER_ROOT = CACHE_ROOT / "03_adapters"
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_SLUG = "Qwen--Qwen3.5-4B"

#: the question-only query draw both inference stages used
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}
EMBEDDER = "a3c1f067c13d3cc8"          # nomic-embed-text-v1.5, search_document
DATASET_EMBEDDER = "b37e80a31644dc03"  # authored by the 8B run; recipe-keyed
SAMP_GREEDY, SAMP_SAMPLED = "6f000f01", "58d3f985"
MAX_NEW_TOKENS = 128

#: Qwen3.5-4B is hybrid: `full_attention_interval: 4`, so every 4th layer is
#: softmax attention and the rest are gated-delta-rule linear attention.
N_LAYERS = 32
FULL_ATTN_LAYERS = [i for i in range(N_LAYERS) if i % 4 == 3]
LINEAR_ATTN_LAYERS = [i for i in range(N_LAYERS) if i % 4 != 3]
ATTN_NUM_HEADS = 16
#: hidden state h_{L+1} is the output of transformer layer L; h0 is the embedding
FULL_ATTN_STATES = [L + 1 for L in FULL_ATTN_LAYERS]
LINEAR_ATTN_STATES = [L + 1 for L in LINEAR_ATTN_LAYERS]
N_STATES = N_LAYERS + 1

METRICS = {"cosine": "cosine", "frobenius": "frobenius",
           "cka": "cka", "bw": "bures_wasserstein"}
METRIC_COLS = list(METRICS)

#: input dim per projection, which is what constrains a BW selection
PROJ_DIN = {"q": 2560, "q_query": 2560, "q_gate": 2560, "k": 2560, "v": 2560,
            "qkv": 2560, "z": 2560, "o": 4096, "out": 4096}

NO_CKA_GROUP = "CKA is single-block:\nno layer grouping"
NO_CKA_ROWS = "CKA needs >1 row"
NO_BW_ROWS = "BW is rank-1 here\n(same as cosine)"


def _bw_mixed(projs) -> str:
    dims = sorted({PROJ_DIN[p] for p in projs})
    return f"BW cannot stack\ninput dims {dims}"


def thirds(xs):
    """Split a layer list into three contiguous, near-equal depth bands."""
    xs = list(xs)
    k = len(xs)
    return xs[: k // 3], xs[k // 3: 2 * k // 3], xs[2 * k // 3:]


# ── Level builders ────────────────────────────────────────────────────────────

def behavioral_cells(idx, ids):
    def sel(replicates, sampling, **over):
        base = {"draw": DRAW, "max_new_tokens": MAX_NEW_TOKENS,
                "replicates": replicates, "sampling_hash": sampling,
                "embedder_hash": EMBEDDER, "replicate_reduction": "all",
                "view": "matrix", "normalize": "none", "representation": "matrix"}
        return dict(base, **over)

    per_query = {"replicate_reduction": "mean", "representation": "matrix",
                 "renormalize": True}
    model_mean = {"replicate_reduction": "mean", "representation": "mean",
                  "renormalize": True}

    rows = {
        # greedy `per query` is omitted, not forgotten: averaging over a single
        # replicate is the identity, so it would duplicate `per generation`.
        "greedy · per generation":  sel(1, SAMP_GREEDY),
        "greedy · model mean":      sel(1, SAMP_GREEDY, **model_mean),
        "R=16 · per generation":    sel(16, SAMP_SAMPLED),
        "R=16 · per query":         sel(16, SAMP_SAMPLED, **per_query),
        "R=16 · model mean":        sel(16, SAMP_SAMPLED, **model_mean),
    }
    cells = {}
    for row, selector in rows.items():
        for col in METRIC_COLS:
            if selector.get("representation") == "mean" and col in ("cka", "bw"):
                cells[(row, col)] = NO_CKA_ROWS if col == "cka" else NO_BW_ROWS
                continue
            cells[(row, col)] = _compute_distance_matrix(
                idx, "behavioral", METRICS[col], ids, behavioral_selector=selector)
    return list(rows), cells


def _fsel(layers):
    return {"draw": DRAW, "mode": "input", "pooling": "mean", "layers": layers,
            "view": "concat", "normalize": "layer", "max_new_tokens": None}


def functional_layer_rows():
    return {
        # h0 is the negative control: LoRA never touches the embedding matrix, so
        # all 16 models are identical here and whatever a metric reports is its
        # own noise floor.
        "h0 · embeddings (control)":  _fsel([0]),
        "h1 · first linear-attn":     _fsel([1]),
        "h4 · first full-attn":       _fsel([4]),
        "h16 · mid-stack (full-attn)": _fsel([16]),
        "h32 · final hidden state":   _fsel([32]),
    }


def functional_group_rows():
    early, mid, late = thirds(range(1, N_STATES))
    return {
        "all 33 layers (reference)": _fsel(None),
        "early third":              _fsel(list(early)),
        "middle third":             _fsel(list(mid)),
        "late third":               _fsel(list(late)),
        "full-attn outputs":        _fsel(FULL_ATTN_STATES),
        "linear-attn outputs":      _fsel(LINEAR_ATTN_STATES),
    }


def functional_cells(idx, ids, rows):
    cells = {}
    for row, selector in rows.items():
        for col in METRIC_COLS:
            cells[(row, col)] = _compute_distance_matrix(
                idx, "functional", METRICS[col], ids, functional_selector=selector)
    return list(rows), cells


def dataset_cells(idx, ids):
    row = "dataset text · mean · n1000_s00"
    cells = {}
    for col in METRIC_COLS:
        if col == "cka":
            cells[(row, col)] = NO_CKA_ROWS
        elif col == "bw":
            cells[(row, col)] = NO_BW_ROWS
        else:
            cells[(row, col)] = _compute_distance_matrix(
                idx, "dataset_embedding", METRICS[col], ids,
                dataset_selector={"n_samples": 1000, "seed": 0,
                                  "representation": "mean"},
                embedder_hash=DATASET_EMBEDDER)
    return [row], cells


# ── Structural ────────────────────────────────────────────────────────────────
#   Built directly on src.notebook.structure, whose four builders are all defined
#   on the LoRA product B @ A and evaluate it through a low-rank identity rather
#   than forming the d×d matrix.

def _structural_dm(weights, names, layers, projs, metric_tag):
    from src.notebook.structure import (
        bures_wasserstein_distance_matrix, cka_distance_matrix,
        cosine_similarity_matrix, frobenius_distance_matrix,
    )
    if metric_tag == "cosine":
        n, S = cosine_similarity_matrix(weights, layers=layers, projections=projs)
        return as_distance_matrix(n, S, "cosine", "structural", similarity=True)
    if metric_tag == "frobenius":
        n, D = frobenius_distance_matrix(weights, layers=layers, projections=projs)
        return as_distance_matrix(n, D, "frobenius", "structural")
    if metric_tag == "bw":
        n, D = bures_wasserstein_distance_matrix(
            weights, layers=layers, projections=projs)
        return as_distance_matrix(n, D, "bures_wasserstein", "structural")
    if metric_tag == "cka":
        if len(layers) != 1 or len(projs) != 1:
            return NO_CKA_GROUP
        n, D = cka_distance_matrix(weights, layers[0], projs[0])
        return as_distance_matrix(n, D, "cka_linear", "structural")
    raise ValueError(metric_tag)


def _load_weights(names, layers, projs):
    from src.notebook.lora_weights import load_lora_weights
    return load_lora_weights(
        names, ADAPTER_ROOT, layer_indices=list(layers), projections=list(projs),
        attn_num_heads=ATTN_NUM_HEADS)


def _structural_grid(names, specs):
    """specs: {row_label: (layers, projections)} -> (rows, cells).

    The weights are read **once** for the union of every row's selection, not once
    per row. Each adapter is ~50 MB and the rows overlap heavily, so per-row
    loading re-read tens of gigabytes and left the job I/O-bound at single-digit
    CPU. The builders already take `layers`/`projections` and intersect against
    what is present, so one collection serves every row.
    """
    all_layers = sorted({int(l) for layers, _ in specs.values() for l in layers})
    all_projs = sorted({p for _, projs in specs.values() for p in projs})
    print(f"    loading {len(names)} adapters × {len(all_projs)} projections once …")
    weights = _load_weights(names, all_layers, all_projs)

    cells = {}
    for row, (layers, projs) in specs.items():
        print(f"    {row}")
        for col in METRIC_COLS:
            if col == "bw" and len({PROJ_DIN[p] for p in projs}) > 1:
                cells[(row, col)] = _bw_mixed(projs)
                continue
            cells[(row, col)] = _structural_dm(
                weights, names, list(layers), list(projs), col)
    return list(specs), cells


def structural_layer_specs():
    # Single (layer, projection) cells, so CKA is available on every one and no
    # BW selection can straddle two input dims. o_proj / out_proj are the two
    # families' output projections, which makes them the comparable pair.
    specs = {}
    for L in (3, 15, 31):
        specs[f"full-attn · layer {L} · o_proj"] = ([L], ["o"])
    for L in (0, 16, 30):
        specs[f"linear-attn · layer {L} · out_proj"] = ([L], ["out"])
    return specs


def structural_group_specs():
    se, sm, sl = thirds(FULL_ATTN_LAYERS)
    le, lm, ll = thirds(LINEAR_ATTN_LAYERS)
    return {
        "all layers · all projections": (range(N_LAYERS),
                                         ["q", "k", "v", "o", "qkv", "z", "out"]),
        "full-attn · q,k,v,o":          (FULL_ATTN_LAYERS, ["q", "k", "v", "o"]),
        "linear-attn · qkv,z,out":      (LINEAR_ATTN_LAYERS, ["qkv", "z", "out"]),
        # dim-pure companions, so BW has a value at family resolution too
        "full-attn · q,k,v (d_in 2560)": (FULL_ATTN_LAYERS, ["q", "k", "v"]),
        "linear-attn · qkv,z (d_in 2560)": (LINEAR_ATTN_LAYERS, ["qkv", "z"]),
        "output projections (d_in 4096)": (range(N_LAYERS), ["o", "out"]),
        "full-attn · early third":      (se, ["q", "k", "v"]),
        "full-attn · middle third":     (sm, ["q", "k", "v"]),
        "full-attn · late third":       (sl, ["q", "k", "v"]),
        "linear-attn · early third":    (le, ["qkv", "z"]),
        "linear-attn · middle third":   (lm, ["qkv", "z"]),
        "linear-attn · late third":     (ll, ["qkv", "z"]),
    }


def structural_projection_specs():
    specs = {
        "full-attn · q_proj (whole)": (FULL_ATTN_LAYERS, ["q"]),
        # attn_output_gate fuses a gate into q_proj, so half its rows are not
        # queries at all. The halves are interleaved per head, not stacked.
        "full-attn · q_proj query half": (FULL_ATTN_LAYERS, ["q_query"]),
        "full-attn · q_proj gate half":  (FULL_ATTN_LAYERS, ["q_gate"]),
        "full-attn · k_proj": (FULL_ATTN_LAYERS, ["k"]),
        "full-attn · v_proj": (FULL_ATTN_LAYERS, ["v"]),
        "full-attn · o_proj": (FULL_ATTN_LAYERS, ["o"]),
        "linear-attn · in_proj_qkv": (LINEAR_ATTN_LAYERS, ["qkv"]),
        "linear-attn · in_proj_z":   (LINEAR_ATTN_LAYERS, ["z"]),
        "linear-attn · out_proj":    (LINEAR_ATTN_LAYERS, ["out"]),
    }
    return specs


# ── Ground truth, for the layer sweep and the cross-level closer ──────────────

def truth_dm(ids):
    from scipy.spatial.distance import pdist, squareform
    W = np.vstack([mixture_weights(m) for m in ids])
    return as_distance_matrix(list(ids), squareform(pdist(W)), "euclidean",
                              taxonomy="ground_truth")


def dcor_vs_truth(dm, tdm):
    """Bias-corrected distance correlation against the ground-truth simplex.

    `distance_correlation` returns a bare float; it is `dcor_test` that returns a
    result object carrying `.statistic`. Note the U-centred dCor* lives on a
    squared scale and may legitimately be negative, so do not clip it.
    """
    from src.analysis.matrices import distance_correlation
    return float(distance_correlation(dm, tdm))


# ── Figures ───────────────────────────────────────────────────────────────────

def emit(level, rows, cells, outdir, title):
    dm_grid(cells, rows, METRIC_COLS, f"{title} — distance matrices",
            savepath=outdir / f"fig_{level}_dm_grid.png")
    plt.close("all")
    mds_grid(cells, rows, METRIC_COLS, f"{title} — MDS embeddings",
             savepath=outdir / f"fig_{level}_mds_grid.png")
    plt.close("all")


def emit_detail(level, row, cells, outdir, title):
    """One annotated panel per metric for the level's reference rung."""
    for col in METRIC_COLS:
        cell = cells.get((row, col))
        if cell is None or isinstance(cell, str):
            continue
        dm_grid({(row, col): cell}, [row], [col],
                f"{title} · {col}", annot=True, panel_w=6.4, panel_h=6.0,
                savepath=outdir / f"fig_{level}_dm_{col}.png")
        plt.close("all")
        mds_grid({(row, col): cell}, [row], [col], f"{title} · {col}",
                 panel_w=5.4, panel_h=5.0,
                 savepath=outdir / f"fig_{level}_mds_{col}.png")
        plt.close("all")


def layer_sweep(idx, ids, outdir):
    """Where in the stack does mixture identity live? One line per metric."""
    tdm = truth_dm(ids)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = list(range(N_STATES))
    for col in METRIC_COLS:
        ys = []
        for h in xs:
            try:
                dm = _compute_distance_matrix(
                    idx, "functional", METRICS[col], ids,
                    functional_selector=_fsel([h]))
                ys.append(dcor_vs_truth(dm, tdm))
            except Exception as exc:
                # h0 is all-zero by construction, so some metrics legitimately
                # have nothing to report there. Say which layer and why rather
                # than letting a NaN propagate silently into the plot.
                print(f"    {col}: h{h} skipped — {type(exc).__name__}: {exc}")
                ys.append(np.nan)
        ax.plot(xs, ys, marker="o", ms=2.6, lw=1.1, label=col)
        if np.isnan(ys).all():
            print(f"    {col}: no finite values across all {len(xs)} layers")
        else:
            print(f"    {col}: best h{int(np.nanargmax(ys))} = {np.nanmax(ys):.3f}")
    # Distinguishable from the whitegrid gridlines, which are also pale grey —
    # at 0.85 these marks were invisible and the legend entry meaningless.
    for h in FULL_ATTN_STATES:
        ax.axvline(h, color="0.62", lw=0.9, ls=(0, (4, 2)), zorder=0)
    ax.axvline(FULL_ATTN_STATES[0], color="0.62", lw=0.9, ls=(0, (4, 2)), zorder=0,
               label="full-attention output")
    ax.set_xlim(-0.5, N_STATES - 0.5)
    ax.set_xlabel("hidden state (h0 = embeddings)")
    ax.set_ylabel("distance correlation vs ground-truth simplex")
    ax.set_title("Functional level — mixture signal by depth", fontsize=10)
    ax.legend(fontsize=6, ncol=5)
    save_figure(fig, str(outdir / "fig_functional_layer_sweep.png"))
    plt.close("all")


def cross_level(best, ids, outdir):
    """One canonical panel per level, plus agreement with the ground truth."""
    tdm = truth_dm(ids)
    rows = ["reference rung"]
    cols = list(best)
    cells = {("reference rung", lvl): dm for lvl, dm in best.items()}
    mds_grid(cells, rows, cols, "Cross-level comparison — MDS (cosine)",
             savepath=outdir / "fig_crosslevel_mds.png")
    plt.close("all")
    dm_grid(cells, rows, cols, "Cross-level comparison — distance matrices (cosine)",
            savepath=outdir / "fig_crosslevel_dm.png")
    plt.close("all")

    lines = ["| level | dCor vs ground truth |", "|---|---|"]
    for lvl, dm in best.items():
        lines.append(f"| {lvl} | {dcor_vs_truth(dm, tdm):.4f} |")
    table = "\n".join(lines)
    (outdir / "crosslevel_agreement.md").write_text(table + "\n")
    print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", action="append",
                    choices=["behavioral", "functional", "structural",
                             "dataset_embedding"],
                    help="restrict to one or more levels (default: all)")
    ap.add_argument("--outdir", default=str(REPO_ROOT / "figures" / "simplex3_qwen"))
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the 33-layer functional sweep")
    ap.add_argument("--skip-detail", action="store_true",
                    help="skip the per-metric detail figures")
    args = ap.parse_args()

    levels = args.level or ["behavioral", "functional", "structural",
                            "dataset_embedding"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_style("two_col_full")

    idx = scan_cache(str(CACHE_ROOT), base_model_id=BASE_MODEL,
                     behavioral_draw=DRAW, functional_draw=DRAW)
    ids = sort_by_mixture(idx.model_ids)
    names = [Path(m).name for m in ids]
    print(f"cache: {CACHE_ROOT}\nmodels: {len(ids)}")
    if len(ids) != 16:
        raise SystemExit(f"expected 16 models, found {len(ids)} — cache incomplete?")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ternary_legend(ax, ids, label_models=True)
    ax.set_title("simplex3 mixtures — barycentric colour key", fontsize=9)
    save_figure(fig, str(outdir / "fig_ternary_legend.png"))
    plt.close("all")

    best = {}

    if "behavioral" in levels:
        print("behavioral …")
        rows, cells = behavioral_cells(idx, ids)
        emit("behavioral", rows, cells, outdir, "Behavioral level")
        if not args.skip_detail:
            emit_detail("behavioral", rows[0], cells, outdir, "Behavioral")
        best["behavioral"] = cells[(rows[0], "cosine")]

    if "functional" in levels:
        print("functional (individual layers) …")
        rows, cells = functional_cells(idx, ids, functional_layer_rows())
        emit("functional_layers", rows, cells, outdir,
             "Functional level — individual layers")
        print("functional (groupings) …")
        grows, gcells = functional_cells(idx, ids, functional_group_rows())
        emit("functional_groups", grows, gcells, outdir,
             "Functional level — layer groupings")
        if not args.skip_detail:
            emit_detail("functional", grows[0], gcells, outdir, "Functional")
        best["functional"] = gcells[(grows[0], "cosine")]
        if not args.skip_sweep:
            print("functional layer sweep …")
            layer_sweep(idx, ids, outdir)

    if "structural" in levels:
        for tag, specs, title in [
            ("structural_layers", structural_layer_specs(),
             "Structural level — individual layers"),
            ("structural_groups", structural_group_specs(),
             "Structural level — layer groupings"),
            ("structural_projections", structural_projection_specs(),
             "Structural level — per projection"),
        ]:
            print(f"{tag} …")
            rows, cells = _structural_grid(names, specs)
            # structure.py returns bare adapter names; relabel to the full ids so
            # the mixture parser and the colour system see what they expect.
            cells = _relabel(cells, names, ids)
            emit(tag, rows, cells, outdir, title)
            if tag == "structural_groups":
                if not args.skip_detail:
                    emit_detail("structural", rows[0], cells, outdir, "Structural")
                best["structural"] = cells[(rows[0], "cosine")]

    if "dataset_embedding" in levels:
        print("dataset_embedding …")
        rows, cells = dataset_cells(idx, ids)
        emit("dataset_embedding", rows, cells, outdir, "Dataset-embedding level")
        if not args.skip_detail:
            emit_detail("dataset_embedding", rows[0], cells, outdir, "Dataset embedding")
        best["dataset_embedding"] = cells[(rows[0], "cosine")]

    if len(best) > 1:
        print("cross-level …")
        cross_level(best, ids, outdir)

    n = len(list(outdir.glob("*.png")))
    print(f"\nwrote {n} figures to {outdir}")


def _relabel(cells, names, ids):
    """Map bare adapter names back onto the full model ids used everywhere else."""
    lookup = dict(zip(names, ids))
    out = {}
    for key, cell in cells.items():
        if isinstance(cell, str):
            out[key] = cell
            continue
        cell.model_ids = [lookup.get(m, m) for m in cell.model_ids]
        out[key] = cell
    return out


if __name__ == "__main__":
    main()
