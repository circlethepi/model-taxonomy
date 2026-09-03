#!/usr/bin/env python
"""The simplex3 cross-level closer for OLMo-2-0425-1B-Instruct.

The smallest base model in the family of suites — 1B against qwen's 4B, llama3i's
8B and nemo's 12B — run with the same sixteen recipes, the same query draw and the
same four levels. That size gap is the reason to want it: it asks whether a level
recovers the mixture simplex because the taxonomy is sound or because the model
was large enough to make almost any read of it work.

**Only the cross-level outputs are written.** This directory holds
``fig_crosslevel_mds.png``, ``fig_crosslevel_dm.png``, their ``_dataset_cosine``
variants, and the two score tables — no per-level grids, no per-metric detail
panels, no layer sweep, no colour key. Read the qwen or llama3i directories for
those; this run exists to sit in the comparison, not to be browsed on its own.

Suppressing the grids does **not** narrow what is computed. Every selected
perspective is still built and scored, so ``crosslevel_agreement.md`` carries the
full per-level ranking and ``crosslevel_scores.csv`` every cell's dCor and
Procrustes residual, exactly as they would in a run that also drew the figures.

What is selected
----------------
A **perspective** is a surrogate together with a metric — one cell of a grid.
This driver draws the same perspectives as ``figures/simplex3_llama3i``, which
are the ten best per level from the qwen run. Both models are uniform softmax
attention with no fused output gate, so llama's unprefixed surrogate names
carry over unchanged at the behavioral, structural and dataset levels.

The functional level is the exception, because two of its row names encode a
position in the stack and this model is 16 layers deep where llama is 32:
``h32 · final hidden state`` and ``all 33 layers (reference)`` become ``h16``
and ``all 17``. Note that ``h16`` here is the *last* hidden state, not the
mid-stack probe that carries that name on the 32-layer suites; both are derived
from :data:`N_LAYERS` below so they cannot disagree with each other, and
:func:`src.plots.simplex_suite.compact_selection` raises on a surrogate the level
does not build, so a wrong count fails loudly rather than quietly drawing
something else.

A 16-layer stack also makes ``late third`` a five-layer selection where llama's
is eleven. That is the honest translation of the surrogate — it names a fraction
of the stack, not a layer count — and it is worth remembering when this run's
functional numbers are read beside the larger models'.

Usage
-----
    python figures/simplex3_olmo2/make_figures.py
    python figures/simplex3_olmo2/make_figures.py --level structural --level behavioral
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

BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: Decoder layers, from the checkpoint's own config. Uniform softmax attention,
#: no `full_attention_interval` and no `attn_output_gate`, so the suite emits
#: this model's surrogates unprefixed and builds no `linear-attn` row.
N_LAYERS = 16

#: The query draw both inference stages used, identical to the qwen, llama3i and
#: nemo runs' -- which is what makes the four comparable. Read off this run's own
#: ``04_activations/**/n100_s01_fea27ccee/`` paths, not assumed.
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}


def _grid(rows, cols):
    """Every (surrogate, metric) pair in a full rectangle."""
    return [(r, c) for r in rows for c in cols]


#: Per level, the perspectives to score. Keep in step with
#: ``figures/simplex3_llama3i/make_figures.py`` and
#: ``figures/simplex3_nemo/make_figures.py``: an edit here that is not mirrored
#: there silently ends the comparison the three runs exist for.
SELECT = {
    # Architecture-free -- these surrogates name a sampling regime and a
    # reduction, so they are identical to the llama3i driver's.
    "behavioral": (
        [("R=16 · per query", m)
         for m in ("cka", "frobenius", "euclidean", "cosine", "bw")]
        + [("greedy · per generation", m)
           for m in ("cosine", "frobenius", "euclidean")]
        + [("R=16 · per generation", m) for m in ("frobenius", "euclidean")]
    ),
    # Three rows, as for llama3i: on a uniform-attention model `full-attn
    # outputs` would select exactly what the reference row selects, so the suite
    # does not build it and asking for it here would plot one row twice.
    "functional": _grid(
        [f"h{N_LAYERS} · final hidden state", "late third",
         f"all {N_LAYERS + 1} layers (reference)"],
        ["cosine", "cka", "frobenius", "euclidean", "bw"],
    ),
    # qwen's top ten with the `full-attn · ` prefixes dropped, since there is
    # one attention family to name. Cosine takes nine of the ten; `late third`
    # is the only surrogate whose frobenius reading also places.
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
    # Three non-transform perspectives exist in total, so "top ten" is all of them.
    "dataset_embedding": [
        ("dataset text · mean · n1000_s00", m)
        for m in ("frobenius", "euclidean", "cosine")
    ],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", action="append", choices=list(SELECT),
                    help="restrict to one or more levels (default: all). The "
                         "cross-level closer needs at least two.")
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute everything, ignoring stored results "
                         "(still writes them back)")
    args = ap.parse_args()

    levels = args.level or list(SELECT)
    suite.run_suite(
        base_model=BASE_MODEL,
        draw=DRAW,
        outdir=args.outdir,
        cache_root=args.cache_root,
        levels=levels,
        no_cache=args.no_cache,
        # The fleet transforms are off, so a `centered` or `whitened` surrogate is
        # never built and none is selected above.
        surrogates=False,
        select={k: v for k, v in SELECT.items() if k in levels},
        crosslevel_only=True,
        source="figures/simplex3_olmo2/make_figures.py",
    )


if __name__ == "__main__":
    main()
