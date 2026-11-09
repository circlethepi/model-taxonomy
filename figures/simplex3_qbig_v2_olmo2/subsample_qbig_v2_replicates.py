#!/usr/bin/env python
"""How many replicates does the simplex need?  qbig_v2 along the replicate axis.

**qbig_v2** -- the standing sixteen-adapter OLMo-2 yahoo simplex read through
a 2000-query uniform yahoo probe, 64 sampled replicates per query at T=1.0.
**R** -- the number of sampled generations per query averaged into the
query's embedding.  See ``docs/terminology.md``.

This script scores **sampled behavioral at T=1.0 only**, at
R = 1, 2, 4, 8, 16, 32, 64, along the query axis.

**Each R is one fixed replicate set, replicates 0..R-1** (the user's call,
2026-11-03) -- the rule ``subsample_qbig_v2_temp.py`` uses for its R=16.  There
are no replicate draws: every trial is a query draw, so the spread is
query-sampling spread alone.  qbig's script crossed ten query draws with ten
replicate draws instead.  Consequences, trial for trial:

* R=64 reproduces ``qbig_v2_subsample.csv``'s sampled rows;
* R=16 reproduces ``qbig_v2_temp_subsample.csv``'s T=1.0 rows.

**The query draws are qbig_v2's**: the same grid, ``N_DRAWS``, seed and draw
order as ``subsample_qbig_v2.py``, plus the full 2000-query pool once.

**The full R=64 matrix is checked against the standard pipeline** before any
subset is scored.  The pipeline has no first-R read, so the R<64 matrices
themselves are not checked; the two reproductions above are checked instead
(``--skip-reproduction-check`` skips them).

**A collapsed trial is scored NaN.**  When all sixteen adapters give the same
representation on a trial's queries (R=1 hit this, job 1474614), Procrustes has
nothing to fit; both scores are written as NaN (the user's call, 2026-11-03),
the row keeps its ``n_queries``, and the log ends with a count per R and n.

Writes ``qbig_v2_replicates.csv`` beside this file, in the columns of
``qbig_v2_subsample.csv``.

Usage::

    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2_replicates.py
    python figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2_replicates.py --quick
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

#: Powers of two up to the full 64; R=1 is one sampled continuation per query.
REPLICATE_GRID = [1, 2, 4, 8, 16, 32, 64]

#: The CSVs whose rows this run must reproduce, and at which R.  Scores are
#: compared to this tolerance: the same arithmetic on the same inputs.
REPRODUCTIONS = {64: ("qbig_v2_subsample.csv", "1.0"),
                 16: ("qbig_v2_temp_subsample.csv", "1.0")}
REPRODUCTION_TOL = 1e-9

#: Procrustes' refusal for a trial whose 16 adapters all coincide.
DEGENERATE = "degenerate configuration: all points coincide"
#: ``(R, n_queries)`` -> collapsed trials, so the run's log says where they fell.
COLLAPSED: dict[tuple, int] = {}


def score_parts(p, idx, ids, key) -> dict:
    """Score one trial; a collapsed one (all 16 adapters coincide, so
    Procrustes has nothing to fit) is recorded as NaN on both scores -- the
    user's call, 2026-11-03 -- and counted under *key*."""
    try:
        return v2.score_parts(p, idx, ids, "behavioral")
    except ValueError as e:
        if str(e) != DEGENERATE:
            raise
        COLLAPSED[key] = COLLAPSED.get(key, 0) + 1
        return {"dcor": float("nan"), "disparity": float("nan")}


def check_reproductions(rows) -> None:
    """Stop if R=64 or R=16 disagrees with the run it should reproduce."""
    ours = {(int(r["replicates"]), int(r["n_queries"]), int(r["query_draw"])):
            (r["dcor"], r["disparity"]) for r in rows}
    for r, (name, temp) in REPRODUCTIONS.items():
        path = HERE / name
        if not path.exists():
            print(f"  reproduction check R={r}: {name} not found, skipped")
            continue
        n = worst = 0
        with path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                if (row["perspective"] != "sampled" or row["temperature"] != temp
                        or int(row["replicates"]) != r):
                    continue
                key = (r, int(row["n_queries"]), int(row["query_draw"]))
                if key not in ours:
                    continue
                for a, b in zip(ours[key], (float(row["dcor"]),
                                            float(row["disparity"]))):
                    # Both collapsed is agreement; one collapsed is not.
                    if np.isnan(a) and np.isnan(b):
                        continue
                    gap = abs(a - b)
                    worst = np.inf if np.isnan(gap) else max(worst, gap)
                n += 1
        if n == 0:
            raise RuntimeError(f"no R={r} rows of {name} matched this run")
        if worst > REPRODUCTION_TOL:
            raise RuntimeError(f"R={r} differs from {name} by {worst:.3g} "
                               f"over {n} trials")
        print(f"  reproduction check R={r}: {n} trials match {name} "
              f"(max gap {worst:.1e})", flush=True)


def run(args) -> None:
    query_grid, n_draws, grid = v2.QUERY_GRID, v2.N_DRAWS, REPLICATE_GRID
    if args.quick:
        # A shape check, not a result.  A trimmed grid draws different query
        # sets from the full one, so there is nothing to reproduce.
        query_grid, n_draws, grid = [2, 1000], 2, [1, 16, 64]
        args.skip_reproduction_check = True

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
    samp = qb.sampling_hash(v2.TEMPERATURE)
    t0 = time.time()

    print(f"loading sampled T={v2.TEMPERATURE} ({samp}) …", flush=True)
    X = qb.load_behavioral(index, ids, emb, replicates=qb.R_TOTAL,
                           sampling_hash=samp)
    qb.check_against_pipeline(
        index, ids, {("behavioral", "sampled"): reduce_replicates(X)}, emb, samp)

    rows = []
    for r in grid:
        p = cosine_parts(reduce_replicates(X, np.arange(r)), ids)
        for n_q, qsets in qdraws.items():
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    embedder=v2.EMBEDDER, level="behavioral",
                    perspective="sampled", label="sampled", layers="",
                    temperature=v2.TEMPERATURE, replicates=r, n_queries=n_q,
                    query_draw=qi, n_query_draws=len(qsets),
                    **score_parts(p, qidx, ids, (r, n_q))))
        print(f"  R={r}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    del X
    # Reported before the reproduction check, so a failed check still says it.
    print(f"\ncollapsed trials (scored NaN): {sum(COLLAPSED.values())}")
    for (r, n_q), c in sorted(COLLAPSED.items()):
        print(f"  R={r:2d}  n={n_q:4d}: {c}")

    if not args.skip_reproduction_check:
        check_reproductions(rows)

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
    p.add_argument("--skip-reproduction-check", action="store_true",
                   help="do not compare R=64 / R=16 against the earlier CSVs")
    p.add_argument("--out", default="qbig_v2_replicates.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
