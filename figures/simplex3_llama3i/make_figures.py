#!/usr/bin/env python
"""The simplex3 suite for Llama-3.1-8B-Instruct, restricted to chosen perspectives.

The instruct sibling of the 8B checkpoint the raw-prompted ``simplex3`` suite
used, and chat-prompted like ``simplex3_qwen``. That is the point of the run:
against qwen it varies model family alone, and against ``simplex3`` it varies
prompting regime alone, where comparing those two directly would confound both.

**Do not confuse this run's cache entries with ``simplex3``'s.** Both live under
``meta-llama--…`` slugs in the shared cache and both carry sixteen ``yahoo_*``
adapters. The instruct run is ``meta-llama--Llama-3.1-8B-Instruct`` and its
adapter names end ``_fea27ccee``, the chat prompt-format id; the base run is
``meta-llama--Llama-3.1-8B`` and its names end ``_b5008``.

What is drawn
-------------
A **perspective** is a surrogate together with a metric — one cell of a grid.
Rather than repeat qwen's full sweep (42/82/84/5 perspectives across the four
levels), this draws the **ten best per level from the qwen run**, which
:mod:`src.plots.simplex_suite` renders as the smallest rectangle covering them.

Carrying a selection across architectures is not a relabelling, and most of
qwen's structural winners do not survive it. Qwen3.5-4B is a hybrid: every fourth
layer is softmax attention and the rest are gated-delta-rule linear attention, and
its q_proj carries a fused output gate. Llama-3.1-8B is uniform softmax with no
gate. So every ``linear-attn · …`` surrogate and both q_proj half-splits have no
llama counterpart at all, and what carries over is the ``full-attn · …`` half,
which the suite emits unprefixed on a uniform model because there is only one
family to name. ``scripts/check_analysis.py`` pins that degradation on both
layouts.

Two levels of this run were unrunnable until 2026-09-02: the functional
activations were never written and two adapters were missing their R=16
generations, because ``04_functional_qonly.sh`` and
``05_behavioral_qonly_shard1.sh`` died in the 2026-08-31 submission on the atomic
temp-name race that ``cd0c62d`` later fixed.

Usage
-----
    python figures/simplex3_llama3i/make_figures.py
    python figures/simplex3_llama3i/make_figures.py --level structural
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plots import simplex_suite as suite  # noqa: E402

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

#: The query draw both inference stages used. Identical to the qwen run's, which
#: is what makes the two comparable at all; read off this run's own
#: ``04_activations/**/runs/*.json`` rather than assumed.
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}


def _grid(rows, cols):
    """Every (surrogate, metric) pair in a full rectangle."""
    return [(r, c) for r in rows for c in cols]


#: Per level, the perspectives to draw.
#:
#: Behavioral, structural and dataset are qwen's top ten by dCor, skipping the
#: fleet-transform surrogates (none wins anywhere) and any surrogate with no
#: llama equivalent. Dataset has only three non-transform perspectives in total,
#: so "top ten" is all of them.
#:
#: Functional is specified directly rather than taken from the ranking. Note what
#: is *not* here: qwen's ranking puts `full-attn outputs` third, but on a
#: uniform-attention model every layer is full-attention, so that surrogate
#: selects exactly what `all 33 layers (reference)` selects. The suite omits it
#: for that reason, and asking for it here would plot one row twice.
SELECT = {
    "behavioral": (
        [("R=16 · per query", m)
         for m in ("cka", "frobenius", "euclidean", "cosine", "bw")]
        + [("greedy · per generation", m)
           for m in ("cosine", "frobenius", "euclidean")]
        + [("R=16 · per generation", m) for m in ("frobenius", "euclidean")]
    ),
    "functional": _grid(
        ["h32 · final hidden state", "late third", "all 33 layers (reference)"],
        ["cosine", "cka", "frobenius", "euclidean", "bw"],
    ),
    # Cosine takes nine of the ten; `late third` is the only surrogate whose
    # frobenius reading also places, which is why the rectangle is 9 x 2 rather
    # than 9 x 1. The eight unselected frobenius cells are drawn because they
    # fall inside it -- they cost a metric each on tensors already read.
    "structural": [
        ("late third", "cosine"),
        ("q_proj (whole)", "cosine"),
        ("all layers · all projections", "cosine"),
        ("q,k,v (dim-pure)", "cosine"),
        ("middle third", "cosine"),
        ("k_proj", "cosine"),
        ("late third", "frobenius"),
        ("v_proj", "cosine"),
        ("output projections", "cosine"),
        ("early third", "cosine"),
    ],
    "dataset_embedding": [
        ("dataset text · mean · n1000_s00", m)
        for m in ("frobenius", "euclidean", "cosine")
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", action="append", choices=list(SELECT),
                    help="restrict to one or more levels (default: all)")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute everything, ignoring stored results "
                         "(still writes them back)")
    ap.add_argument("--skip-detail", action="store_true")
    args = ap.parse_args()

    levels = args.level or list(SELECT)
    suite.run_suite(
        base_model=BASE_MODEL,
        draw=DRAW,
        outdir=args.outdir,
        cache_root=args.cache_root,
        levels=levels,
        skip_detail=args.skip_detail,
        no_cache=args.no_cache,
        # The fleet transforms are off, so a `centered` or `whitened` surrogate is
        # never built and none is selected above.
        surrogates=False,
        select={k: v for k, v in SELECT.items() if k in levels},
        source="figures/simplex3_llama3i/make_figures.py",
    )


if __name__ == "__main__":
    main()
