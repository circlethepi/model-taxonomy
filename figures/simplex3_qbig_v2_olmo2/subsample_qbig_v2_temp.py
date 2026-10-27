#!/usr/bin/env python
"""Does the probe-depth curve depend on temperature?  qbig_v2 at ten temperatures.

**qbig_v2_temp** -- the 2026-10-27 extension of qbig_v2: the same sixteen
adapters, the same 2000-query uniform yahoo probe and the same R=64 decode,
repeated at T = 0.1, 0.2, ..., 0.9 on l40s only.  T=1.0 and greedy are qbig_v2's
own decodes.  See ``docs/terminology.md``.

This script scores **behavioral only**, at **R=16**, along the query axis:

* sampled at T = 0.1 ... 0.9 (the qbig_v2_temp tree) and T = 1.0 (qbig_v2);
* greedy (qbig_v2), one continuation per query.

**R=16 is one fixed replicate set, replicates 0-15** (the user's call,
2026-10-27), the same sixteen at every temperature.  There are no replicate
draws: every trial is a query draw, so the spread is query-sampling spread
alone.  qbig's script crossed ten query draws with ten replicate draws instead.

**The query draws are qbig_v2's**: the same grid, the same ``N_DRAWS``, the
same seed and the same draw order as ``subsample_qbig_v2.py``, so every row
here reads exactly the query set of the matching qbig_v2 row, trial for trial.

**Each temperature's full R=64 matrix is checked against the standard
pipeline** before its R=16 subset is scored.  The pipeline has no first-16
read, so the R=16 matrices themselves are not checked.

Writes ``qbig_v2_temp_subsample.csv`` beside this file, in the columns of
``qbig_v2_subsample.csv``.

Usage::

    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2_temp.py
    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2_temp.py --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex3_qbig_olmo2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

# Importing subsample_qbig_v2 points qbig's loaders at the qbig_v2 draw.
import subsample_qbig_v2 as v2  # noqa: E402
import subsample_qbig as qb  # noqa: E402
from src.analysis.subsample import QueryPool, cosine_parts, reduce_replicates  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402

HERE = Path(__file__).resolve().parent

#: 0.1 ... 0.9 from the qbig_v2_temp tree, 1.0 from qbig_v2.
TEMPERATURES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

#: The fixed replicate set: the first sixteen of the sixty-four decoded.
REPLICATES = np.arange(16)


def run(args) -> None:
    query_grid, n_draws, temperatures = v2.QUERY_GRID, v2.N_DRAWS, TEMPERATURES
    if args.quick:
        # A shape check, not a result.
        query_grid, n_draws, temperatures = [2, 1000], 2, [1.0]

    suite.apply_architecture(suite.architecture(qb.BASE_MODEL))
    index, ids = qb.load_index()
    print(f"models: {len(ids)}", flush=True)

    # Drawn exactly as subsample_qbig_v2.py draws them, so the sets match.
    pool = QueryPool.from_draw(qb.CACHE_ROOT, v2.DRAW)
    rng_q = np.random.default_rng(args.seed)
    qdraws = {n_q: pool.draw_queries(n_q, n_draws, rng_q, by=None,
                                     composition=None)
              for n_q in query_grid}
    qdraws[pool.n] = [np.arange(pool.n)]

    emb = qb.EMBEDDERS[v2.EMBEDDER]
    t0 = time.time()
    rows = []

    def score(persp, temp, r, p):
        for n_q, qsets in qdraws.items():
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    embedder=v2.EMBEDDER, level="behavioral", perspective=persp,
                    label=persp, layers="", temperature=temp, replicates=r,
                    n_queries=n_q, query_draw=qi, n_query_draws=len(qsets),
                    **v2.score_parts(p, qidx, ids, "behavioral")))
        print(f"  {persp} T={temp}: {len(rows)} rows, {time.time() - t0:.0f}s",
              flush=True)

    for temp in temperatures:
        samp = qb.sampling_hash(temp)
        print(f"loading sampled T={temp} ({samp}) …", flush=True)
        X = qb.load_behavioral(index, ids, emb, replicates=qb.R_TOTAL,
                               sampling_hash=samp)
        qb.check_against_pipeline(
            index, ids, {("behavioral", "sampled"): reduce_replicates(X)}, emb, samp)
        p = cosine_parts(reduce_replicates(X, REPLICATES), ids)
        del X
        score("sampled", temp, len(REPLICATES), p)

    print("loading greedy …", flush=True)
    G = qb.unit_rows(qb.load_behavioral(index, ids, emb, replicates=1,
                                        sampling_hash=suite.SAMP_GREEDY)[:, :, 0, :])
    qb.check_against_pipeline(index, ids, {("behavioral", "greedy"): G}, emb)
    score("greedy", "", 1, cosine_parts(G, ids))

    out = HERE / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=v2.FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="query-draw seed (0 matches subsample_qbig_v2.py)")
    p.add_argument("--quick", action="store_true",
                   help="trim the grid to a shape check, not a result")
    p.add_argument("--out", default="qbig_v2_temp_subsample.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
