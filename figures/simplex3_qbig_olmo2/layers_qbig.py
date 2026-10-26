#!/usr/bin/env python
"""Which hidden states carry the simplex?  Score qbig's functional level per layer.

**qbig** -- the 2026-09-17 tree that reads the standing sixteen-adapter OLMo-2
yahoo simplex through a *deep* probe of 1000 queries.  See
``docs/terminology.md``.

``subsample_qbig.py`` scores the functional level through two surrogates, all
hidden states and the final hidden state.  This script scores it through every
single hidden state and through a handful of layer combinations, each along the
same query-count axis, so a layer's curve answers "how many queries does this
layer need" in the same units as the figures already drawn.

**The surrogates.**  Every one is the standing functional selector -- input
mode, mean pooling over the prompt's tokens, layer-normalized concatenation,
cosine -- with only ``layers`` changed.  OLMo-2-1B has 16 decoder layers, so
17 hidden states, ``h0`` (the embeddings) through ``h16`` (the final state):

* **single layers** -- each of ``h1`` … ``h16`` alone.  ``h0`` is not scored:
  LoRA never touches the embedding matrix, so all sixteen models are identical
  there, every distance is 0, and neither score is defined (Procrustes stops on
  a configuration whose points all coincide).  It stays inside "all hidden
  states", as it does in the standing row.
* **thirds** -- ``early third`` (h1-h5), ``middle third`` (h6-h10), ``late
  third`` (h11-h16): the suite's ``thirds()`` split of h1-h16, under the labels
  ``simplex_suite.functional_group_rows`` already uses.
* **every Nth layer** -- one layer from each evenly spaced step of h1-h16,
  joined.  Both ends are kept, so the steps are the divisors of 15: every 15th
  (h1, h16), every 5th (h1, h6, h11, h16), every 3rd (h1, h4, … h16) and every
  layer (h1-h16, i.e. all hidden states but ``h0``).
* **all hidden states** -- h0-h16, the standing row, as the reference.

**Every combination is a mean of per-layer parts, exactly.**  Under
``normalize="layer"`` each layer block is scaled to unit length per query
before the blocks are joined, and the joined row is renormalized.  A row built
from *k* unit blocks has norm sqrt(k), so the inner product of two such rows is
the mean of their per-layer inner products.  :class:`CosineParts` keeps exactly
those inner products per query, so each layer is decomposed once and every
combination is an average of seventeen arrays of shape ``(16, 16, 1000)`` --
no concatenated matrix is ever built.  This holds only while no block has a
zero row (the cache's normalize leaves those at zero, which breaks the sqrt(k));
the script stops if it finds one.

**Two cross-checks before any score is written.**  At the full probe the
all-hidden-states and final-hidden-state matrices built from parts are compared
with (a) ``build_taxonomy_artifacts``, the path every other figure takes, and
(b) the same surrogates read through ``ActivationCache.load``, as
``subsample_qbig.py`` reads them.  The query sets are drawn with the same seed
and calls as ``subsample_qbig.py``, so the all-states and final-state rows here
must equal that CSV's functional rows trial for trial; ``make_qbig_layer_figures``
does not re-check this, so run ``--compare`` once after a rescore.

Writes ``qbig_layers.csv`` beside this file.

Usage::

    python figures/simplex3_qbig_olmo2/layers_qbig.py
    python figures/simplex3_qbig_olmo2/layers_qbig.py --compare   # vs qbig_subsample.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import subsample_qbig as qb  # noqa: E402
from src.analysis.subsample import CosineParts, QueryPool, cosine_parts  # noqa: E402
from src.cache.activation_cache import ActivationCache  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402

HERE = Path(__file__).resolve().parent

#: CSV columns, one row per (surrogate, query draw).  ``layers`` is the
#: space-separated hidden-state list, so a row is self-describing whatever its
#: label says; ``layer_set`` is what the figure groups by.
FIELDS = ["perspective", "label", "layer_set", "layers", "n_layers",
          "n_queries", "query_draw", "n_query_draws", "dcor", "disparity"]


def surrogates(n_states: int) -> list[tuple[str, str, str, list[int]]]:
    """``(perspective, label, layer_set, layers)`` for every surrogate scored.

    Derived from the hidden-state count rather than written down, so the
    "every Nth" steps are the ones that divide this model's h1..h(L) evenly.
    """
    top = n_states - 1                                  # h16 on OLMo-2-1B
    span = list(range(1, n_states))                     # h1..h16
    # h0 is left out: identical across the fleet, so no score exists for it.
    out = [(f"h{ell}", f"h{ell}", "single", [ell]) for ell in span]
    early, mid, late = suite.thirds(span)
    out += [("early_third", "early third", "third", list(early)),
            ("middle_third", "middle third", "third", list(mid)),
            ("late_third", "late third", "third", list(late))]
    for step in sorted(s for s in range(1, top) if (top - 1) % s == 0):
        layers = list(range(1, n_states, step))
        name = "every layer" if step == 1 else f"every {_ordinal(step)} layer"
        span_txt = (f"h1–h{top}" if step == 1
                    else ", ".join(f"h{ell}" for ell in layers))
        out.append((f"every_{step}", f"{name} ({span_txt})", "every_nth", layers))
    out.append(("all_states", f"all hidden states (h0–h{top})", "all",
                list(range(n_states))))
    return out


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ── Per-layer parts ──────────────────────────────────────────────────────────

def layer_parts(index, ids, states) -> dict[int, CosineParts]:
    """``{layer: CosineParts}`` of each hidden state, unit-scaled per query.

    One layer of all sixteen models is in memory at a time: ``(16, 1000, 2048)``
    float64, 260 MB.
    """
    cache = ActivationCache(qb.CACHE_ROOT)
    by_id = {e.model_id: e for e in index.entries}
    parts = {}
    for ell in states:
        blocks = []
        for m in ids:
            e = by_id[m]
            X = cache.load_layers(e.base_model_id, e.model_id, qb.DRAW,
                                  mode="input", pooling="mean", layers=[ell],
                                  max_new_tokens=None)[ell].astype(np.float64)
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            if np.any(norms < 1e-12):
                raise RuntimeError(
                    f"{m} h{ell}: {int(np.sum(norms < 1e-12))} zero rows. The "
                    f"mean-of-parts shortcut assumes every block is unit length; "
                    f"read this surrogate through ActivationCache.load instead.")
            blocks.append(X / norms)
        parts[ell] = cosine_parts(np.stack(blocks), ids)
        print(f"  h{ell}: decomposed", flush=True)
    return parts


def combine(parts: dict[int, CosineParts], layers: list[int], ids) -> CosineParts:
    """The parts of the layer-normalized concatenation of *layers*."""
    num = np.mean([parts[ell].num for ell in layers], axis=0)
    sq = np.mean([parts[ell].sq for ell in layers], axis=0)
    return CosineParts(num=num, sq=sq, model_ids=list(ids))


def score_parts(p: CosineParts, idx, ids) -> dict:
    dm = p.distance_matrix(idx, taxonomy="functional")
    dcor, disp = suite.truth_scores(dm, ids)
    return {"dcor": float(dcor), "disparity": float(disp)}


# ── Cross-checks ─────────────────────────────────────────────────────────────

def check(index, ids, parts, states) -> None:
    """Stop unless the parts-built full-probe matrices match both other reads."""
    built = {("functional", "all_states"): combine(parts, states, ids),
             ("functional", "final_state"): combine(parts, [states[-1]], ids)}
    specs = qb.pipeline_specs(qb.EMBEDDERS[qb.DEFAULT_EMBEDDERS[0]])
    for key, p in built.items():
        mine = p.distance_matrix(None, taxonomy="functional").matrix
        theirs = np.asarray(qb.pipeline_matrix(index, ids, specs[key]).matrix)
        layers = None if key[1] == "all_states" else [states[-1]]
        direct = cosine_parts(qb.load_functional(index, ids, layers), ids
                              ).distance_matrix(None, taxonomy="functional").matrix
        for name, other in (("pipeline", theirs), ("cache.load", direct)):
            gap = float(np.max(np.abs(mine - other)))
            print(f"  check {key[1]} vs {name}: max |gap| = {gap:.2e}", flush=True)
            if gap > qb.PIPELINE_TOL:
                raise RuntimeError(
                    f"{key[1]}: the parts-built matrix differs from the {name} "
                    f"read by {gap:.2e} (> {qb.PIPELINE_TOL:.0e}).")


def compare(out: Path) -> None:
    """Report the largest gap between this CSV and ``qbig_subsample.csv``."""
    def read(path, keep):
        with path.open() as fh:
            return {(r["perspective"], int(r["n_queries"]), int(r["query_draw"])):
                    (float(r["dcor"]), float(r["disparity"]))
                    for r in csv.DictReader(fh) if keep(r)}
    ref = read(HERE / "qbig_subsample.csv", lambda r: r["level"] == "functional")
    top = max(int(k[0][1:]) for k in read(out, lambda r: r["layer_set"] == "single"))
    mine = read(out, lambda r: r["perspective"] in ("all_states", f"h{top}"))
    mine = {(("final_state" if p == f"h{top}" else p), n, d): v
            for (p, n, d), v in mine.items()}
    missing = set(ref) ^ set(mine)
    if missing:
        raise RuntimeError(f"{len(missing)} trials in one CSV and not the other, "
                           f"e.g. {sorted(missing)[:3]}")
    gap = max(max(abs(a - b) for a, b in zip(ref[k], mine[k])) for k in ref)
    print(f"{len(ref)} trials matched; max |gap| vs qbig_subsample.csv = {gap:.2e}")


# ── The sweep ────────────────────────────────────────────────────────────────

def run(args) -> None:
    suite.apply_architecture(suite.architecture(qb.BASE_MODEL))
    index, ids = qb.load_index()
    states = qb.stored_states(index, ids)
    if states != list(range(suite.N_STATES)):
        raise RuntimeError(f"stored hidden states {states} are not "
                           f"h0..h{suite.N_STATES - 1}")
    print(f"models: {len(ids)}; hidden states h0..h{states[-1]}", flush=True)

    t0 = time.time()
    parts = layer_parts(index, ids, states)
    check(index, ids, parts, states)

    # Same seed, same calls, same order as subsample_qbig.py's query draws, so a
    # trial here and a trial there read the same queries.
    pool = QueryPool.from_draw(qb.CACHE_ROOT, qb.DRAW)
    rng_q = np.random.default_rng(args.seed)
    qdraws = {n_q: pool.draw_queries(n_q, 1 if n_q == pool.n else qb.N_DRAWS,
                                     rng_q, by=None, composition=None)
              for n_q in qb.QUERY_GRID}

    rows = []
    for persp, label, layer_set, layers in surrogates(len(states)):
        p = combine(parts, layers, ids)
        for n_q, qsets in qdraws.items():
            for qi, qidx in enumerate(qsets):
                rows.append(dict(
                    perspective=persp, label=label, layer_set=layer_set,
                    layers=" ".join(map(str, layers)), n_layers=len(layers),
                    n_queries=n_q, query_draw=qi, n_query_draws=len(qsets),
                    **score_parts(p, qidx, ids)))
        print(f"  {label}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

    out = HERE / args.out
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out} in {time.time() - t0:.0f}s")
    compare(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0,
                   help="query-draw seed; must match subsample_qbig.py's")
    p.add_argument("--out", default="qbig_layers.csv")
    p.add_argument("--compare", action="store_true",
                   help="only compare an existing CSV with qbig_subsample.csv")
    args = p.parse_args()
    if args.compare:
        compare(HERE / args.out)
    else:
        run(args)


if __name__ == "__main__":
    main()
