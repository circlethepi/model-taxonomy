#!/usr/bin/env python
"""How does diluting the probe move the simplex?  qmix, composed from pools.

**qmix_v2** -- the 2026-10-29 rebuild of **qmix** (query mixture: the yahoo
percentage of the probe, the rest a diluting corpus the adapters never saw).
The standing sixteen-adapter OLMo-2 yahoo simplex is held fixed; only the probe
varies.  See ``docs/terminology.md``.

qmix decoded a separate 100-row probe at every (yahoo %, seed).  qmix_v2
decodes each corpus ONCE, as a pool, and composes every cell from the pools by
subsampling, the way ``subsample_qbig_v2.py`` reads probe depth:

* **yahoo** -- qbig_v2's 2000-row uniform draw (seed 2, format ``ea27ccee``),
  decoded at R=64; replicates 0-15 are read, so it scores at R=16;
* **dolly** -- 2000 rows of the dolly arm's 0%-yahoo recipe, seed 0, R=16;
* **oasst1cjk** -- all 859 rows of ``oasst1_pairs_cjk`` (zh 768, ja 86, ko 5),
  seed 0, R=16.

The pools render under different prompt formats (the dolly and CJK pools under
qmix's union format, ``631eeef5``), but a yahoo row renders byte-identically
under both -- checked before scoring (``--skip-render-check`` skips it).  All
were decoded on l40s at T=1.0, batch 2, generation seed 0, and embedded with
nomic-embed-text-v2-moe.

**One cell** is ``(arm, yahoo %, n)``.  Each of its ``N_DRAWS`` trials is drawn
independently:

* ``k ~ Binomial(n, p)`` yahoo rows -- each query is yahoo with probability p,
  so the composition itself varies trial to trial (the user's call,
  2026-10-29); at 0% and 100% it is fixed by construction;
* ``k`` rows uniformly without replacement from the yahoo pool, and ``n - k``
  from the arm's diluent pool.

Levels are 0, 1, 2, 5, 10, 20, 50 per arm, and 100 once, shared by both arms
(arm ``yahoo``).  Query counts are ``QUERY_GRID`` up to 1000 for dolly and
yahoo, up to 500 for oasst1cjk -- plus one oasst1cjk point at the whole 859-row
pool and 0% yahoo, scored once (``n_query_draws = 1``).

**Composition is exact.**  Every perspective is a cosine over per-query rows,
which :class:`CosineParts` keeps as per-query sums, so a pool pair is two parts
concatenated along the query axis and a trial is one index set into it.  The
functional reads are means of per-layer parts, as in ``layers_qbig.py``.

**Every perspective reads the same trials**, so two perspectives differ trial
by trial only in the representation: the nine qbig_v2 reads (sampled, greedy,
functional all states, the three thirds, h1/h9/h16), each scored by Procrustes
disparity (0 = perfect) and dCor* (1 = perfect) against the fixed truth
simplex.

**Each pool's full matrices are checked against the standard pipeline** before
any trial is scored, and the run stops if they disagree beyond round-off.

Writes ``qmix_v2_subsample.csv`` beside this file.

Usage::

    python figures/simplex3_qmix_v2_olmo2/subsample_qmix_v2.py
    python figures/simplex3_qmix_v2_olmo2/subsample_qmix_v2.py --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex3_qbig_olmo2"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex3_qbig_v2_olmo2"))

import numpy as np  # noqa: E402

import subsample_qbig as qb  # noqa: E402
import layers_qbig as lq  # noqa: E402
from subsample_qbig_v2 import functional_reads, score_parts  # noqa: E402
from src.analysis.subsample import CosineParts, cosine_parts, reduce_replicates  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402

HERE = Path(__file__).resolve().parent

#: The three pools, as cache draw keys.  ``r`` is the R each was decoded at;
#: ``read`` is how many replicates are scored (the first ``read``).
POOLS = {
    "yahoo": {"draw": {"recipe_hash": "075110fbec1c029d", "n_samples": 2000,
                       "seed": 2, "prompt_format_id": "ea27ccee"},
              "r": 64, "read": 16},
    "dolly": {"draw": {"recipe_hash": "8681960d93a5b6ea", "n_samples": 2000,
                       "seed": 0, "prompt_format_id": "631eeef5"},
              "r": 16, "read": 16},
    "oasst1cjk": {"draw": {"recipe_hash": "45f02ffe0e5cc3e2", "n_samples": 859,
                           "seed": 0, "prompt_format_id": "631eeef5"},
                  "r": 16, "read": 16},
}

#: The configs whose draws the render check re-derives: one per format.
QBIG_V2_CONFIG = (qb.REPO / "experiments" / "simplex3_qbig_v2_olmo2_qbig_v2"
                  / "behavioral_qonly_shard0.yaml")
QMIX_V2_CONFIG = (qb.REPO / "experiments" / "simplex3_qmix_v2_olmo2"
                  / "functional_qmix_yahoo000_dolly100_qtc_s00.yaml")

EMBEDDER = "v2moe"
TEMPERATURE = 1.0
R_READ = 16

#: Yahoo percentages per arm; 100 is scored once, as arm ``yahoo``.
YAHOO_PCTS = [0, 1, 2, 5, 10, 20, 50]
QUERY_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
#: The largest query count each arm is subsampled at.
ARM_MAX_N = {"dolly": 1000, "oasst1cjk": 500, "yahoo": 1000}
N_DRAWS = 100

FIELDS = ["embedder", "level", "perspective", "label", "layers",
          "temperature", "replicates", "arm", "yahoo_pct", "n_queries",
          "n_yahoo", "query_draw", "n_query_draws", "dcor", "disparity"]


# ── Pools ────────────────────────────────────────────────────────────────────

def concat_parts(a: CosineParts, b: CosineParts) -> CosineParts:
    """Two pools' parts as one pool: *a*'s queries first, then *b*'s."""
    return CosineParts(num=np.concatenate([a.num, b.num], axis=2),
                       sq=np.concatenate([a.sq, b.sq], axis=1),
                       model_ids=list(a.model_ids))


def load_pool(name, index, ids, states) -> dict:
    """``{read: CosineParts}`` for one pool, after checking it against the pipeline.

    qbig's loaders read the draw and R from module globals, so they are pointed
    at this pool for the duration.  The sampled check runs at the R the pool was
    decoded at (the pipeline has no entry at any other R), and the read then
    keeps the first ``read`` replicates.
    """
    pool = POOLS[name]
    qb.DRAW, qb.R_TOTAL = pool["draw"], pool["r"]
    emb = qb.EMBEDDERS[EMBEDDER]
    samp = qb.sampling_hash(TEMPERATURE)
    print(f"\n── pool {name}: {pool['draw']} ──", flush=True)

    X = qb.load_behavioral(index, ids, emb, replicates=pool["r"], sampling_hash=samp)
    qb.check_against_pipeline(index, ids, {("behavioral", "sampled"): reduce_replicates(X)},
                              emb, samp)
    V = reduce_replicates(X[:, :, :pool["read"], :])
    del X
    G = qb.unit_rows(qb.load_behavioral(index, ids, emb, replicates=1,
                                        sampling_hash=suite.SAMP_GREEDY)[:, :, 0, :])
    qb.check_against_pipeline(index, ids, {("behavioral", "greedy"): G}, emb)
    out = {"sampled": cosine_parts(V, ids), "greedy": cosine_parts(G, ids)}
    del V, G

    layers = lq.layer_parts(index, ids, states)
    lq.check(index, ids, layers, states)
    out["layers"] = layers
    return out


def combine_pools(yahoo: dict, dil: dict) -> dict:
    return {"sampled": concat_parts(yahoo["sampled"], dil["sampled"]),
            "greedy": concat_parts(yahoo["greedy"], dil["greedy"]),
            "layers": {ell: concat_parts(yahoo["layers"][ell], dil["layers"][ell])
                       for ell in yahoo["layers"]}}


def check_yahoo_render() -> None:
    """Stop unless qbig_v2's yahoo rows render identically under qmix's format.

    The yahoo pool was decoded under the stock format and the diluent pools
    under the union one; composing them is only sound if a yahoo row's prompt
    is the same string under both.
    """
    import copy

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from _utils import make_queries

    cfg = yaml.safe_load(QBIG_V2_CONFIG.read_text())
    union = yaml.safe_load(QMIX_V2_CONFIG.read_text())["prompt_format"]
    q, key, idx = make_queries(cfg)
    cfg2 = copy.deepcopy(cfg)
    cfg2["prompt_format"] = union
    q2, key2, idx2 = make_queries(cfg2)
    same = sum(a == b for a, b in zip(q, q2))
    print(f"render check: {same}/{len(q)} yahoo prompts identical under "
          f"{key['prompt_format_id']} and {key2['prompt_format_id']}", flush=True)
    if idx != idx2 or same != len(q) or key["recipe_hash"] != POOLS["yahoo"]["draw"]["recipe_hash"]:
        raise RuntimeError("the yahoo pool does not render identically under "
                           "qmix_v2's prompt format; the pools cannot be composed.")


# ── Trials ───────────────────────────────────────────────────────────────────

def draw_trials(rng, n_pool: dict) -> list[tuple]:
    """Every trial, as ``(arm, pct, n, draw, n_draws, k, idx)``.

    *idx* indexes the yahoo pool followed by the arm's diluent pool; arm
    ``yahoo`` (the 100% level) indexes the yahoo pool alone.
    """
    ny = n_pool["yahoo"]
    trials = []
    for arm in ("dolly", "oasst1cjk"):
        nd = n_pool[arm]
        for pct in YAHOO_PCTS:
            for n in [q for q in QUERY_GRID if q <= ARM_MAX_N[arm]]:
                for qi in range(N_DRAWS):
                    k = int(rng.binomial(n, pct / 100))
                    y = rng.choice(ny, size=k, replace=False)
                    d = rng.choice(nd, size=n - k, replace=False)
                    trials.append((arm, pct, n, qi, N_DRAWS, k,
                                   np.concatenate([y, ny + d])))
    # The whole CJK pool, undiluted, once.
    nd = n_pool["oasst1cjk"]
    trials.append(("oasst1cjk", 0, nd, 0, 1, 0, ny + np.arange(nd)))
    for n in [q for q in QUERY_GRID if q <= ARM_MAX_N["yahoo"]]:
        for qi in range(N_DRAWS):
            trials.append(("yahoo", 100, n, qi, N_DRAWS, n,
                           rng.choice(ny, size=n, replace=False)))
    return trials


def run(args) -> None:
    global QUERY_GRID, N_DRAWS
    if args.quick:
        # A shape check, not a result.
        QUERY_GRID = [2, 500]
        N_DRAWS = 2

    if not args.skip_render_check:
        check_yahoo_render()

    suite.apply_architecture(suite.architecture(qb.BASE_MODEL))
    index, ids = qb.load_index()
    print(f"models: {len(ids)}", flush=True)
    states = qb.stored_states(index, ids)
    if states != list(range(suite.N_STATES)):
        raise RuntimeError(f"stored hidden states {states} are not "
                           f"h0..h{suite.N_STATES - 1}")

    t0 = time.time()
    pools = {name: load_pool(name, index, ids, states) for name in POOLS}
    n_pool = {name: POOLS[name]["draw"]["n_samples"] for name in POOLS}
    composed = {"yahoo": pools["yahoo"],
                "dolly": combine_pools(pools["yahoo"], pools["dolly"]),
                "oasst1cjk": combine_pools(pools["yahoo"], pools["oasst1cjk"])}
    del pools

    trials = draw_trials(np.random.default_rng(args.seed), n_pool)
    print(f"\n{len(trials)} trials, {time.time() - t0:.0f}s", flush=True)

    reads = [("behavioral", "sampled", "sampled", "", TEMPERATURE, R_READ, None),
             ("behavioral", "greedy", "greedy", "", "", 1, None)]
    reads += [("functional", persp, label, layers, "", 0, layers)
              for persp, label, layers in functional_reads(len(states))]

    rows = []
    for level, persp, label, layers, temp, r, fl in reads:
        if fl is None:
            parts = {arm: c[persp] for arm, c in composed.items()}
        else:
            parts = {arm: lq.combine(c["layers"], fl, ids)
                     for arm, c in composed.items()}
        for arm, pct, n, qi, nq, k, idx in trials:
            rows.append(dict(
                embedder=EMBEDDER if level == "behavioral" else "n/a",
                level=level, perspective=persp, label=label,
                layers=" ".join(map(str, layers)) if layers else "",
                temperature=temp, replicates=r, arm=arm, yahoo_pct=pct,
                n_queries=n, n_yahoo=k, query_draw=qi, n_query_draws=nq,
                **score_parts(parts[arm], idx, ids, level)))
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
    p.add_argument("--seed", type=int, default=0, help="trial-draw seed")
    p.add_argument("--quick", action="store_true",
                   help="trim the grid to a shape check, not a result")
    p.add_argument("--skip-render-check", action="store_true",
                   help="skip re-deriving the yahoo draw under both formats")
    p.add_argument("--out", default="qmix_v2_subsample.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
