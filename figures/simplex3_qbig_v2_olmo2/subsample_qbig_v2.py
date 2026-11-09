#!/usr/bin/env python
"""How much probe does the simplex need?  Subsample qbig_v2 along the query axis.

**qbig_v2** -- the 2026-10-27 re-run of qbig: the standing sixteen-adapter
OLMo-2 yahoo simplex read through a 2000-query probe drawn *uniformly* from
yahoo (one pooled entry over topics 0-9, not a third per group), seed 2, 64
sampled replicates at T=1.0, plus greedy and input-mode functional.  See
``docs/terminology.md``.

The tree is decoded once at full depth.  This script asks what a shallower
probe would have reported, by subsampling the cached artifacts along the
**query axis only**: for each query count in ``QUERY_GRID``, ``N_DRAWS`` query
sets are drawn uniformly without replacement from the 2000-query pool, and the
full pool is scored once as a single point (``n_query_draws = 1``).

**Replicates are held at the full R=64** (the user's call, 2026-10-27).  qbig's
script crossed ten query draws with ten replicate draws per R; here every trial
is a query draw, so the spread is query-sampling spread alone.  A replicate
axis can be added later from the same cache without re-decoding.

**Every perspective reads the same query sets**, so two perspectives differ
trial by trial only in the representation:

* behavioral / sampled -- T=1.0, R=64, replicate mean then renormalized;
* behavioral / greedy -- one continuation per query;
* functional / all_states -- h0..h16, the standing row;
* functional / thirds -- ``1–5``, ``6–10``, ``11–16`` (``suite.thirds``);
* functional / single layers -- h1, h9, h16.

Functional combinations are means of per-layer ``CosineParts``, as in
``layers_qbig.py``, which is exact under ``normalize="layer"``.

**Two scores per trial**, both against the one fixed truth simplex through
``simplex_suite.truth_scores``: Procrustes disparity (0 = perfect) and dCor*
(1 = perfect).

**The full-pool matrices are checked against the standard pipeline** before
any subset is scored -- sampled, greedy, all states and the final state through
``build_taxonomy_artifacts`` -- and the run stops if they disagree beyond
float32 round-off.

Writes ``qbig_v2_subsample.csv`` beside this file.

Usage::

    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2.py
    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2.py --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex3_qbig_olmo2"))

import numpy as np  # noqa: E402

import subsample_qbig as qb  # noqa: E402
import layers_qbig as lq  # noqa: E402
from src.analysis.subsample import QueryPool, cosine_parts, reduce_replicates  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The qbig_v2 probe.  The recipe is one pooled yahoo entry over topics 0-9 at
#: weight 100, which is exactly qmix's undiluted 100%-yahoo recipe (same hash);
#: the draw differs from qmix's by n and from qbig's by recipe.
DRAW = {"recipe_hash": "075110fbec1c029d", "n_samples": 2000, "seed": 2,
        "prompt_format_id": "ea27ccee"}

#: qbig's loaders, pipeline specs and functional decomposition all read the
#: draw from their module global, so pointing it here is what makes every one
#: of them read qbig_v2 instead.  Done once, at import, before anything loads.
qb.DRAW = DRAW

#: v2-moe is the only embedder qbig_v2 was embedded with.
EMBEDDER = "v2moe"
TEMPERATURE = 1.0

#: {1,2,5}x10^k up to 1000; the full 2000-query pool is appended as one point.
QUERY_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
N_DRAWS = 100

FIELDS = ["embedder", "level", "perspective", "label", "layers",
          "temperature", "replicates", "n_queries", "query_draw",
          "n_query_draws", "dcor", "disparity"]


def functional_reads(n_states: int) -> list[tuple[str, str, list[int]]]:
    """``(perspective, label, layers)`` for every functional read scored."""
    top = n_states - 1                                   # h16 on OLMo-2-1B
    out = [("all_states", f"all (0–{top})", list(range(n_states)))]
    for third in suite.thirds(list(range(1, n_states))):
        layers = list(third)
        out.append((f"layers_{layers[0]}_{layers[-1]}",
                    f"{layers[0]}–{layers[-1]}", layers))
    for ell in (1, 9, top):
        out.append((f"h{ell}", f"h{ell}", [ell]))
    return out


def score_parts(parts, idx, ids, taxonomy) -> dict:
    dm = parts.distance_matrix(idx, taxonomy=taxonomy)
    dcor, disp = suite.truth_scores(dm, ids)
    return {"dcor": float(dcor), "disparity": float(disp)}


def run(args) -> None:
    global QUERY_GRID, N_DRAWS
    if args.quick:
        # A shape check, not a result.
        QUERY_GRID = [2, 1000]
        N_DRAWS = 2

    suite.apply_architecture(suite.architecture(qb.BASE_MODEL))
    index, ids = qb.load_index()
    print(f"models: {len(ids)}", flush=True)

    states = qb.stored_states(index, ids)
    if states != list(range(suite.N_STATES)):
        raise RuntimeError(f"stored hidden states {states} are not "
                           f"h0..h{suite.N_STATES - 1}")

    pool = QueryPool.from_draw(qb.CACHE_ROOT, DRAW)
    rng_q = np.random.default_rng(args.seed)
    qdraws = {n_q: pool.draw_queries(n_q, N_DRAWS, rng_q, by=None,
                                     composition=None)
              for n_q in QUERY_GRID}
    qdraws[pool.n] = [np.arange(pool.n)]

    emb = qb.EMBEDDERS[EMBEDDER]
    samp = qb.sampling_hash(TEMPERATURE)
    t0 = time.time()

    # ── Behavioral: each read reduced to (M, Q, d), then decomposed once. ──
    print(f"loading sampled T={TEMPERATURE} ({samp}) …", flush=True)
    X = qb.load_behavioral(index, ids, emb, replicates=qb.R_TOTAL,
                           sampling_hash=samp)
    V = reduce_replicates(X)
    del X
    print("loading greedy …", flush=True)
    G = qb.unit_rows(qb.load_behavioral(index, ids, emb, replicates=1,
                                        sampling_hash=suite.SAMP_GREEDY)[:, :, 0, :])
    qb.check_against_pipeline(index, ids, {("behavioral", "sampled"): V}, emb, samp)
    qb.check_against_pipeline(index, ids, {("behavioral", "greedy"): G}, emb)
    behavioral = [("sampled", TEMPERATURE, qb.R_TOTAL, cosine_parts(V, ids)),
                  ("greedy", "", 1, cosine_parts(G, ids))]
    del V, G

    # ── Functional: per-layer parts, checked against the pipeline. ──
    parts = lq.layer_parts(index, ids, states)
    lq.check(index, ids, parts, states)

    rows = []
    for persp, temp, r, p in behavioral:
        for n_q, qsets in qdraws.items():
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    embedder=EMBEDDER, level="behavioral", perspective=persp,
                    label=persp, layers="", temperature=temp, replicates=r,
                    n_queries=n_q, query_draw=qi, n_query_draws=len(qsets),
                    **score_parts(p, qidx, ids, "behavioral")))
        print(f"  {persp}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

    for persp, label, layers in functional_reads(len(states)):
        p = lq.combine(parts, layers, ids)
        for n_q, qsets in qdraws.items():
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    embedder="n/a", level="functional", perspective=persp,
                    label=label, layers=" ".join(map(str, layers)),
                    temperature="", replicates=0, n_queries=n_q,
                    query_draw=qi, n_query_draws=len(qsets),
                    **score_parts(p, qidx, ids, "functional")))
        print(f"  {label}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0, help="query-draw seed")
    p.add_argument("--quick", action="store_true",
                   help="trim the grid to a shape check, not a result")
    p.add_argument("--out", default="qbig_v2_subsample.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
