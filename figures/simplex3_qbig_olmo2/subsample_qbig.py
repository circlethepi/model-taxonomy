#!/usr/bin/env python
"""How much probe does the simplex need?  Subsample qbig along both probe axes.

**qbig** -- the 2026-09-17 tree that reads the standing sixteen-adapter OLMo-2
yahoo simplex through a *deep* probe: 1000 queries, 64 sampled replicates each.
See ``docs/terminology.md``.

The tree was run once at full depth.  This script asks what a *shallower* probe
would have reported, by subsampling the cached artifacts along two axes:

* **queries** -- how many of the 1000 prompts the taxonomy is read through;
* **replicates** -- how many of the 64 generations per prompt are averaged
  before the query's embedding is formed.

Nothing is re-generated and nothing is re-embedded.  Both reductions are exact
rather than approximate, because the cosine metric flattens its inputs: a
distance over a query subset is a masked sum of per-query parts, and a distance
over a replicate subset is the same pipeline with a different mean.  See
``src/analysis/subsample`` for the decomposition and the checks behind that
claim.

**The draw design is nested, as specified.**  For each query count, ten query
sets are drawn without replacement from the 1000-prompt pool.  Then, *for each
of those query sets*, ten replicate sets are drawn for each replicate count.
So a ``(n_queries, replicates)`` cell holds 100 trials -- ten replicate draws
inside each of ten query draws -- and the spread across them is what the figures
band.

**Two cells are single points, not bands.**  At ``n_queries = 1000`` there is
exactly one subset of the pool without replacement, and at ``replicates = 64``
exactly one subset of the replicates.  Those are drawn once, and carry
``n_draws = 1`` in the CSV so the plotting side can render them as points rather
than as zero-width bands.

**Sampled behavioral is read at every temperature on disk.**  The qbigtemp tree
(2026-10-13) decoded the same draw at T = 0.25, 0.5, 0.75 and 1.0, each its own
cache entry, all on one GPU architecture.  Each temperature runs the full
``(n_queries, replicates)`` grid over the **same** query subsets and the same
replicate subsets, so two temperatures differ trial by trial only in T.  The
T=1.0 entry is qbigtemp's l40s decode; qbig's original mixed-GPU T=1.0 entry
was moved to ``results/archive/qbig_t10_mixed_gpu/`` and is not read.

**Greedy decoding has no replicate axis.**  Greedy is deterministic, so the tree
stores one generation per query; it appears on the query axis only, as its own
perspective.

**Two scores per trial**, both against the one fixed truth simplex through
``simplex_suite.truth_scores``: dCor* (1 = perfect) and Procrustes disparity
(0 = perfect) -- the same pair ``qmix_scores.csv`` carries.

**The full-probe cell is checked against the standard pipeline.**  Before the
sweep, each perspective's ``(1000 queries, R=64)`` distance matrix is rebuilt
through ``build_taxonomy_artifacts`` from the canonical perspective definitions
-- the path every other figure in this project takes -- and the run stops if the
two disagree beyond float32 round-off.  That is what licenses reading every
smaller cell as "what the pipeline would have reported at this depth".

Writes ``qbig_subsample.csv`` beside this file; ``make_qbig_figures.py`` draws it.

Usage::

    python figures/simplex3_qbig_olmo2/subsample_qbig.py
    python figures/simplex3_qbig_olmo2/subsample_qbig.py --embedders v15 v2moe
    python figures/simplex3_qbig_olmo2/subsample_qbig.py --by group
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402

from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.subsample import (  # noqa: E402
    QueryPool, cosine_parts, draw_replicates, reduce_replicates,
)
from src.cache.activation_cache import ActivationCache  # noqa: E402
from src.cache.generated_text_cache import GeneratedTextCache  # noqa: E402
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import raw_mixture_pcts, sort_by_mixture  # noqa: E402

# The canonical perspectives are imported, not copied, as sweep_qmix.py does:
# they are the standing per-level defaults, and the cross-check below is only
# meaningful if it reads the same definitions every other figure reads.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import _META_KEYS, canonical_perspectives  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy")
CACHE_ROOT = REPO / "results" / "shared_cache"
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: The qbig probe.  ``seed=2`` is not incidental -- seeds 0 and 1 on this recipe
#: are the 1000-row training draw and the 100-query canonical probe, so a
#: 1000-query probe had to take the next seed to avoid colliding with the
#: training rows it is meant to be independent of.
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 1000, "seed": 2,
        "prompt_format_id": "ea27ccee"}
MAX_NEW_TOKENS = 128
R_TOTAL = 64

#: Both embedders the tree's generations are embedded under.  ``v15`` is the standing one
#: every other figure in this project uses; ``v2moe`` is the second read, which
#: costs minutes rather than GPU-hours because it re-embeds cached text rather
#: than re-decoding it.
EMBEDDERS = {
    "v15": "a3c1f067c13d3cc8",      # nomic-embed-text-v1.5, search_document
    "v2moe": "0b579825f703fb21",    # nomic-embed-text-v2-moe
}

#: Ten sizes, log-spaced as {1,2,5}x10^k, ending at the full pool.
QUERY_GRID = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
#: Powers of two up to the full replicate count.
REPLICATE_GRID = [2, 4, 8, 16, 32, 64]
N_DRAWS = 10

#: The sampling temperatures the qbigtemp tree decoded.  Every other sampling
#: field is the tree's: top_p 1.0, no top_k, generation seed 0.
TEMPERATURES = [0.25, 0.5, 0.75, 1.0]


def sampling_hash(temperature: float) -> str:
    """The cache's sampling hash for qbigtemp's decode at *temperature*.

    Computed through ``GeneratedTextCache.sampling_hash`` rather than listed, so
    a temperature that was never decoded fails at load instead of reading
    nothing; T=1.0 is checked against the suite's ``SAMP_SAMPLED``.
    """
    h = GeneratedTextCache.sampling_hash({
        "do_sample": True, "temperature": float(temperature), "top_p": 1.0,
        "top_k": None, "generation_seed": 0})
    if temperature == 1.0 and h != suite.SAMP_SAMPLED:
        raise RuntimeError(f"T=1.0 hashes to {h}, not the suite's "
                           f"{suite.SAMP_SAMPLED}; the sampling key changed.")
    return h

#: The four training fields that separate qbig's fleet from everything else in
#: the shared cache.  The mixture filter alone is not enough: the nsweep's
#: n=10000 adapters, the 1004-model group-size pool and the qmix fleet all carry
#: the same corpus, base model and mixtures, and scanning without this pin
#: returns 1700 models instead of 16.
TRAIN_PIN = {"n_samples": 1000, "seed": 0, "lora_rank": 16, "lora_init_seed": 0}

#: The figure reads v2-moe alone (the user's call, 2026-10-09); v1.5 stays
#: reachable with ``--embedders``.
DEFAULT_EMBEDDERS = ["v2moe"]

#: Largest elementwise gap allowed between this script's full-probe distance
#: matrix and the pipeline's.  The cache stores float32 and the pipeline does
#: part of its arithmetic there, where this script works in float64 throughout;
#: cosine distances are O(1e-2), so 1e-5 is round-off and anything above it is a
#: different computation.
PIPELINE_TOL = 1e-5

#: CSV columns, one row per trial.  ``temperature`` is blank for greedy and
#: functional rows, which have none.
FIELDS = ["embedder", "level", "perspective", "temperature", "n_queries",
          "replicates",
          "query_draw", "rep_draw", "n_query_draws", "n_rep_draws", "dcor",
          "disparity"]


# ── Loading ──────────────────────────────────────────────────────────────────

def load_index() -> tuple[CacheIndex, list[str]]:
    wanted = {tuple(m) for m in SPECS["yahoo"].mixture_pcts()}
    index = scan_cache(CACHE_ROOT, base_model_id=BASE_MODEL, datasets=["yahoo"])
    kept = [e for e in index.entries
            if raw_mixture_pcts(e.model_id) in wanted
            and e.n_samples == TRAIN_PIN["n_samples"]
            and e.seed == TRAIN_PIN["seed"]
            and e.lora_rank == TRAIN_PIN["lora_rank"]
            and e.lora_init_seed == TRAIN_PIN["lora_init_seed"]]
    index = CacheIndex(kept, index.cache_root)
    ids = sort_by_mixture(list(index.model_ids))
    if len(ids) != 16:
        raise RuntimeError(
            f"expected the 16-adapter yahoo simplex, found {len(ids)}. The "
            f"training pin {TRAIN_PIN} no longer separates qbig's fleet.")
    return index, ids


def load_behavioral(index, ids, embedder_hash, *, replicates, sampling_hash):
    """``(M, Q, R, d)`` for sampled, ``(M, Q, 1, d)`` for greedy.

    The stored matrix is query-major -- rows ``q0r0 … q0r(R-1)`` are adjacent --
    so the reshape below is the replicate axis and not a transpose of it.
    """
    cache = GeneratedTextCache(CACHE_ROOT)
    by_id = {e.model_id: e for e in index.entries}
    out = []
    for k, m in enumerate(ids):
        e = by_id[m]
        X, _ = cache.load_matrix(
            e.base_model_id, e.model_id, DRAW,
            max_new_tokens=MAX_NEW_TOKENS, replicates=replicates,
            sampling_hash=sampling_hash, embedder_hash=embedder_hash)
        n_q = DRAW["n_samples"]
        if X.shape[0] != n_q * replicates:
            raise RuntimeError(
                f"{m}: stored matrix has {X.shape[0]} rows, expected "
                f"{n_q}x{replicates}")
        out.append(X.reshape(n_q, replicates, X.shape[1]))
        print(f"    behavioral {k + 1:2d}/{len(ids)}", flush=True)
    return np.stack(out)


def load_functional(index, ids, layers):
    """``(M, Q, d)`` concatenated hidden states, layer-normalized.

    ``normalize="layer"`` scales each layer block *within a row*, so it is
    independent of which queries are kept -- which is what makes a query subset
    of this matrix the same thing the pipeline would have built from a smaller
    draw.
    """
    cache = ActivationCache(CACHE_ROOT)
    by_id = {e.model_id: e for e in index.entries}
    out = []
    for k, m in enumerate(ids):
        e = by_id[m]
        rep = cache.load(e.base_model_id, e.model_id, DRAW,
                         mode="input", pooling="mean", layers=layers,
                         view="concat", normalize="layer", max_new_tokens=None)
        out.append(np.asarray(rep.matrix, dtype=np.float64))
        print(f"    functional {k + 1:2d}/{len(ids)}", flush=True)
    return np.stack(out)


def stored_states(index, ids) -> list[int]:
    cache = ActivationCache(CACHE_ROOT)
    e = {x.model_id: x for x in index.entries}[ids[0]]
    return sorted(cache.list_layers(e.base_model_id, e.model_id, DRAW,
                                    "input", "mean", None))


# ── Scoring ──────────────────────────────────────────────────────────────────

def score(V, ids, taxonomy) -> dict:
    """``{dcor, disparity}`` of *V*'s cosine matrix against the truth simplex.

    ``V`` is ``(M, n, d)`` and already reduced -- the replicate mean is taken
    and the query rows renormalized before this point, matching what
    ``build_taxonomy_artifacts`` hands the metric.
    """
    dm = cosine_parts(V, ids).distance_matrix(None, taxonomy=taxonomy)
    dcor, disp = suite.truth_scores(dm, ids)
    return {"dcor": float(dcor), "disparity": float(disp)}


def unit_rows(V):
    """Each query row scaled to unit length -- the greedy read's renormalize."""
    return V / np.clip(np.linalg.norm(V, axis=-1, keepdims=True), 1e-10, None)


# ── The cross-check against the standard pipeline ────────────────────────────

def pipeline_specs(embedder_hash, samp_hash=suite.SAMP_SAMPLED) -> dict:
    """The canonical perspectives this script reproduces, pointed at qbig.

    Only the draw, the embedder and (for sampled behavioral) R and the sampling
    hash change; every other selector field is the canonical one.  Keyed by
    this script's own ``(level, perspective)`` names.
    """
    specs = canonical_perspectives()
    out = {}
    for (level, persp), name, extra in [
            (("functional", "all_states"), "functional_all", {}),
            (("functional", "final_state"), "functional_last", {}),
            (("behavioral", "sampled"), "behavioral",
             {"replicates": R_TOTAL, "embedder_hash": embedder_hash,
              "sampling_hash": samp_hash}),
            (("behavioral", "greedy"), "behavioral_greedy",
             {"embedder_hash": embedder_hash})]:
        spec = specs[name]
        key = f"{spec['taxonomy']}_selector"
        out[(level, persp)] = {**spec, key: {**spec[key], "draw": DRAW, **extra}}
    return out


def pipeline_matrix(index, ids, spec):
    kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
    dm, _ = build_taxonomy_artifacts(
        index, spec["taxonomy"], spec["metric"], cache_root=CACHE_ROOT,
        n_components=(2,), use_cache=True, id_scheme="model_id",
        label="qbig full probe", **kwargs)
    return dm.reindex(list(ids))


def check_against_pipeline(index, ids, full, embedder_hash,
                           samp_hash=suite.SAMP_SAMPLED):
    """Stop unless each full-probe matrix here equals the pipeline's.

    *full* maps ``(level, perspective)`` to this script's ``(M, Q, d)`` read at
    the full probe; a key it lacks is skipped (the functional reads are loaded
    under the first embedder only, and the sampled read once per temperature).
    Returns ``{(level, perspective): max |gap|}``.
    """
    gaps = {}
    for key, spec in pipeline_specs(embedder_hash, samp_hash).items():
        if key not in full:
            continue
        mine = cosine_parts(full[key], ids).distance_matrix(
            None, taxonomy=spec["taxonomy"]).matrix
        theirs = np.asarray(pipeline_matrix(index, ids, spec).matrix)
        gap = float(np.max(np.abs(mine - theirs)))
        gaps[key] = gap
        print(f"  pipeline check {key[0]}/{key[1]}: max |gap| = {gap:.2e}",
              flush=True)
        if gap > PIPELINE_TOL:
            raise RuntimeError(
                f"{key}: the subsampled full-probe matrix differs from the "
                f"pipeline's by {gap:.2e} (> {PIPELINE_TOL:.0e}). The masked-sum "
                f"shortcut no longer reproduces what the pipeline computes; fix "
                f"that before reading any smaller cell.")
    return gaps


# ── The sweep ────────────────────────────────────────────────────────────────

def run(args) -> None:
    global QUERY_GRID, REPLICATE_GRID, N_DRAWS
    if args.quick:
        # A shape check, not a result: enough of the grid to exercise every
        # branch (both ends of both axes, greedy, both functional rows) in
        # about a minute.
        QUERY_GRID = [2, 1000]
        REPLICATE_GRID = [2, 64]
        N_DRAWS = 2

    # canonical_perspectives() places "the final state" from the suite's layer
    # count, which defaults to a 32-layer model; OLMo-2-1B has 16.
    suite.apply_architecture(suite.architecture(BASE_MODEL))

    index, ids = load_index()
    print(f"models: {len(ids)}", flush=True)

    pool = QueryPool.from_draw(CACHE_ROOT, DRAW)
    if args.by == "topic":
        pool = pool.with_topics()
    by = None if args.by == "none" else args.by

    states = stored_states(index, ids)
    print(f"stored hidden states: {states[0]}..{states[-1]} ({len(states)})",
          flush=True)
    if states[-1] != suite.N_STATES - 1:
        raise RuntimeError(
            f"the cache's final hidden state is h{states[-1]} but the suite "
            f"places it at h{suite.N_STATES - 1}; the final-state read and its "
            f"pipeline check would name different layers.")

    rows = []
    t_start = time.time()

    for name in args.embedders:
        emb = EMBEDDERS[name]
        print(f"\n=== embedder {name} ({emb}) ===", flush=True)

        # Query sets are drawn once per embedder pass, from an RNG seeded from
        # the run seed, so a rerun reproduces them and two embedders see the
        # *same* subsets.  Every temperature and every perspective below reads
        # these same sets, which is what makes them comparable trial by trial.
        rng_q = np.random.default_rng(args.seed)
        qdraws = {n_q: pool.draw_queries(n_q, 1 if n_q == pool.n else N_DRAWS,
                                         rng_q, by=by, composition=None)
                  for n_q in QUERY_GRID}

        # Sampled behavioral, one temperature at a time: one (M, Q, R, d) read
        # in memory at once.  The replicate RNG is re-seeded per temperature
        # and consumed in the same order, so each temperature draws the same
        # replicate subsets as the others.
        for T in args.temperatures:
            samp = sampling_hash(T)
            print(f"  loading sampled T={T} ({samp}) …", flush=True)
            X = load_behavioral(index, ids, emb, replicates=R_TOTAL,
                                sampling_hash=samp)
            check_against_pipeline(
                index, ids, {("behavioral", "sampled"): reduce_replicates(X)},
                emb, samp)
            rng_r = np.random.default_rng([args.seed, 1])
            for n_q, qsets in qdraws.items():
                n_qd = len(qsets)
                for qi, qidx in enumerate(qsets):
                    Xq = X[:, qidx]                 # (M, n_q, R, d)
                    for r in REPLICATE_GRID:
                        n_rd = 1 if r == R_TOTAL else N_DRAWS
                        for ri in range(n_rd):
                            ridx = (None if r == R_TOTAL
                                    else draw_replicates(R_TOTAL, r, 1, rng_r)[0])
                            V = reduce_replicates(Xq, ridx, renormalize=True)
                            rows.append(dict(
                                embedder=name, level="behavioral",
                                perspective="sampled", temperature=T,
                                n_queries=n_q, replicates=r, query_draw=qi,
                                rep_draw=ri, n_query_draws=n_qd,
                                n_rep_draws=n_rd,
                                **score(V, ids, "behavioral")))
                print(f"  T={T} n_queries={n_q:4d}: {len(rows)} rows, "
                      f"{time.time() - t_start:.0f}s elapsed", flush=True)
            del X, Xq

        print("  loading greedy …", flush=True)
        G = load_behavioral(index, ids, emb, replicates=1,
                            sampling_hash=suite.SAMP_GREEDY)[:, :, 0, :]

        # The functional level does not read the embedder at all -- it is
        # hidden states, not generated text -- so it is loaded and scored once,
        # under the first embedder pass, and tagged with embedder "n/a".
        if name == args.embedders[0]:
            print("  loading functional (all states) …", flush=True)
            F_all = load_functional(index, ids, None)
            print("  loading functional (final state) …", flush=True)
            F_last = load_functional(index, ids, [states[-1]])
        else:
            F_all = F_last = None

        full = {("behavioral", "greedy"): unit_rows(G)}
        if F_all is not None:
            full[("functional", "all_states")] = F_all
            full[("functional", "final_state")] = F_last
        check_against_pipeline(index, ids, full, emb)
        del full

        for n_q, qsets in qdraws.items():
            n_qd = len(qsets)
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    embedder=name, level="behavioral", perspective="greedy",
                    temperature="", n_queries=n_q, replicates=1,
                    query_draw=qi, rep_draw=0, n_query_draws=n_qd,
                    n_rep_draws=1,
                    **score(unit_rows(G[:, qidx]), ids, "behavioral")))
                if F_all is not None:
                    for persp, F in (("all_states", F_all),
                                     ("final_state", F_last)):
                        rows.append(dict(
                            embedder="n/a", level="functional",
                            perspective=persp, temperature="", n_queries=n_q,
                            replicates=0, query_draw=qi, rep_draw=0,
                            n_query_draws=n_qd, n_rep_draws=1,
                            **score(F[:, qidx], ids, "functional")))
        print(f"  greedy + functional: {len(rows)} rows, "
              f"{time.time() - t_start:.0f}s elapsed", flush=True)

        del G, F_all, F_last

    out = HERE / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out} in {time.time() - t_start:.0f}s")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--embedders", nargs="+", default=DEFAULT_EMBEDDERS,
                   choices=list(EMBEDDERS))
    p.add_argument("--by", default="none", choices=["none", "group", "topic"],
                   help="stratify query draws by this label axis "
                        "(default: uniform over the whole pool)")
    p.add_argument("--temperatures", nargs="+", type=float,
                   default=TEMPERATURES,
                   help="sampled-behavioral temperatures to read "
                        "(default: all four qbigtemp decoded)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true",
                   help="trim the grids to a shape check, not a result")
    p.add_argument("--out", default="qbig_subsample.csv")
    run(p.parse_args())


if __name__ == "__main__":
    main()
