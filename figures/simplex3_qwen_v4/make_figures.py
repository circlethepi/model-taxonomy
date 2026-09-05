#!/usr/bin/env python
"""The simplex3 suite for Qwen3.5-4B, restricted to match the llama3i figures.

Same run as ``figures/simplex3_qwen_v3/``, same numbers, narrower grids. It
exists so the qwen and Llama-3.1-8B-Instruct figure sets can be read side by
side: v3 draws every perspective, the llama set draws ten per level, and two
figures of different widths are hard to compare by eye.

``figures/simplex3_qwen_v3/`` is **not** superseded. It stays the full-width
reference and the regression baseline — every perspective here also appears
there, so their ``matrix_sha256`` values must agree cell for cell, which is what
proves the restriction only selected and never recomputed.

The qwen half of the correspondence
-----------------------------------
Behavioral and dataset surrogates are architecture-free and identical to the
llama driver's. The other two differ, because Qwen3.5-4B is a hybrid — every
fourth layer softmax attention, the rest gated-delta-rule linear attention, and
a fused output gate on q_proj — while Llama-3.1-8B is uniform softmax. So the
suite prefixes qwen's surrogates by family and emits llama's bare:

===========================================  ==============================
qwen (here)                                  llama3i
===========================================  ==============================
``full-attn · late third``                   ``late third``
``full-attn · middle third``                 ``middle third``
``full-attn · early third``                  ``early third``
``full-attn · q_proj (whole)``               ``q_proj (whole)``
``full-attn · k_proj``                       ``k_proj``
``full-attn · v_proj``                       ``v_proj``
``full-attn · q,k,v (d_in 2560)``            ``q,k,v (dim-pure)``
``output projections (d_in 4096)``           ``output projections``
``all layers · all projections``             *(same label)*
``full-attn outputs``                        *(no counterpart)*
===========================================  ==============================

The last row is the asymmetry worth keeping rather than smoothing away: on a
uniform model every layer is full-attention, so ``full-attn outputs`` selects
what ``all 33 layers (reference)`` already selects and the suite omits it. The
functional grid is therefore **4 x 5 here and 3 x 5 for llama**, and that extra
row is the architecture difference made visible.

Usage
-----
    python figures/simplex3_qwen_v4/make_figures.py
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

BASE_MODEL = "Qwen/Qwen3.5-4B"

#: Which corpus this driver plots. Not optional once a second dataset is trained
#: on the same base model: `03_adapters/<base_slug>` holds every adapter for that
#: model regardless of what it was trained on -- the draw arguments set
#: availability flags and never filter -- so an unfiltered scan would return both
#: corpora and `n_expected` would trip.
DATASETS = ["yahoo"]

DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}


def _grid(rows, cols):
    """Every (surrogate, metric) pair in a full rectangle."""
    return [(r, c) for r in rows for c in cols]


#: The llama driver's selection, expressed in qwen's surrogate names. Keep the
#: two in step: an edit here that is not mirrored in
#: ``figures/simplex3_llama3i/make_figures.py`` silently ends the comparison.
SELECT = {
    # Identical to the llama driver -- these surrogates name a sampling regime
    # and a reduction, neither of which is architectural.
    "behavioral": (
        [("R=16 · per query", m)
         for m in ("cka", "frobenius", "euclidean", "cosine", "bw")]
        + [("greedy · per generation", m)
           for m in ("cosine", "frobenius", "euclidean")]
        + [("R=16 · per generation", m) for m in ("frobenius", "euclidean")]
    ),
    # Four rows to llama's three: `full-attn outputs` is a real, distinct
    # surrogate here and definitionally absent there.
    "functional": _grid(
        ["h32 · final hidden state", "late third", "full-attn outputs",
         "all 33 layers (reference)"],
        ["cosine", "cka", "frobenius", "euclidean", "bw"],
    ),
    "structural": [
        ("full-attn · late third", "cosine"),
        ("full-attn · q_proj (whole)", "cosine"),
        ("all layers · all projections", "cosine"),
        ("full-attn · q,k,v (d_in 2560)", "cosine"),
        ("full-attn · middle third", "cosine"),
        ("full-attn · k_proj", "cosine"),
        ("full-attn · late third", "frobenius"),
        ("full-attn · v_proj", "cosine"),
        ("output projections (d_in 4096)", "cosine"),
        ("full-attn · early third", "cosine"),
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
        surrogates=False,
        select={k: v for k, v in SELECT.items() if k in levels},
        datasets=DATASETS,
        source="figures/simplex3_qwen_v4/make_figures.py",
    )


if __name__ == "__main__":
    main()
