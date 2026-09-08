#!/usr/bin/env python
"""The simplex3 cross-level closer for Qwen3.5-4B on databricks-dolly-15k.

The same four levels and the same cross-level closer as the yahoo suites, over a
different corpus. The vertex axis here is instruction category, four groups of two -- g1 classification+closed_qa, g2 summarization+brainstorming, g3 information_extraction+general_qa, g4 creative_writing+open_qa -- so this is a
**3-simplex**: 35 recipes at every 25% composition of four groups, against
yahoo's 16 over three.

What changes at four vertices
-----------------------------
Two things, and neither is cosmetic.

There is **no ternary panel**. `_bary_to_xy` maps to a triangle and
`ternary_legend` labels three corners; a tetrahedron has no honest 2-D
barycentric picture, so the legend is skipped rather than projected. The MDS
panels remain -- they were always 2-D projections of a higher-dimensional
embedding -- and points are coloured by their full 4-vector, so two recipes
differing only in g4 are visibly different.

The **Procrustes disparity is swept over `d = 2 .. K-1`** rather than reported at
a hardcoded 2. `truth_geometry` builds the ground truth in `K-1` dimensions, so a
4-vertex truth is genuinely 3-D; fitting the taxonomy side flat at 2-D and
comparing the two inflates the disparity by all the truth variance outside the
best-fit plane. `crosslevel_scores.csv` carries `procrustes_d2` and
`procrustes_d3` here where the yahoo runs carry `procrustes_d2` alone.

`context` is empty for 100% of g4 and present in ~40-45% of the
other three, so a mixture's g4 weight moves the row *shape* as well as
the task. That is a real property of dolly, not a sampling artefact,
and it is the first thing to suspect if g4 separates unusually well.

**Only the cross-level outputs are written**, as in the yahoo per-model drivers:
the two score tables and the cross-level figures, no per-level grids and no
per-metric detail panels.

Usage
-----
    python figures/simplex3_dolly_qwen/make_figures.py
    python figures/simplex3_dolly_qwen/make_figures.py --level structural --level behavioral
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

#: Which corpus this driver plots, and the reason the argument exists at all:
#: `03_adapters/Qwen--Qwen3.5-4B` holds every adapter trained on this base model
#: whatever corpus it came from -- that is what makes the cache shared -- and the
#: draw arguments set availability flags rather than filtering. Without this the
#: scan returns yahoo's 16 alongside these 35 and the count guard trips.
DATASETS = ["dolly"]

#: 35 recipes: the compositions of 4 into 4 non-negative parts, C(7,3).
N_EXPECTED = 35

#: Decoder layers, from the checkpoint's own config.
N_LAYERS = 32

#: The query draw both inference stages used: the even mixture at n=100, seed 1,
#: question-only, rendered through this corpus's chat projection. Read off the
#: recipe the build job wrote, not assumed -- the projection differs per corpus,
#: so the format id is not the yahoo suites'.
DRAW = {"recipe_hash": "9eccebfa74184124", "n_samples": 100, "seed": 1,
        "prompt_format_id": "f5444e61"}

#: The embedder both new corpora were built with. nomic-embed-text-v2-moe is
#: multilingual, which oasst1 requires -- its vertices are languages, and a
#: monolingual embedder would make that geometry an artefact of the measuring
#: instrument. dolly uses it too so the two corpora are mutually comparable from
#: the start. yahoo's v1.5 hashes stay the suite defaults; these override them.
EMBEDDER = "0b579825f703fb21"
DATASET_EMBEDDER = "a831922612fddb96"


def _grid(rows, cols):
    """Every (surrogate, metric) pair in a full rectangle."""
    return [(r, c) for r in rows for c in cols]


#: Per level, the perspectives to score.
#:
#: These are the yahoo suites' selections held unchanged, and that is the point:
#: a perspective that recovers a topic simplex and fails on a task or language
#: one is a finding, and it is only a finding if the two runs asked for the same
#: cells. Re-tune only after the first full run, and say so here when you do.
SELECT = {
    "behavioral": (
        [("R=16 · per query", m)
         for m in ("cka", "frobenius", "euclidean", "cosine", "bw")]
        + [("greedy · per generation", m)
           for m in ("cosine", "frobenius", "euclidean")]
        + [("R=16 · per generation", m) for m in ("frobenius", "euclidean")]
    ),
    "functional": _grid(
        # Four rows to the uniform-attention drivers' three: `full-attn outputs` is
        # a real, distinct surrogate on a hybrid stack and definitionally absent
        # on the others.
        [f"h{N_LAYERS} · final hidden state", "late third", "full-attn outputs",
         f"all {N_LAYERS + 1} layers (reference)"],
        ["cosine", "cka", "frobenius", "euclidean", "bw"],
    ),
    # Hybrid attention renames this level. Qwen3.5-4B interleaves 8 full-attention
    # layers with 24 linear-attention ones, so a bare "late third" is ambiguous --
    # there are two families and two late thirds -- and `structural_group_specs`
    # prefixes every family-scoped row `full-attn · ` / `linear-attn · `. Two rows
    # are renamed outright as well, because the group is defined by the input
    # dimension the two families happen to share: `q,k,v (dim-pure)` is
    # `full-attn · q,k,v (d_in 2560)`, and `output projections` is
    # `output projections (d_in 4096)`, which is o_proj *and* the linear-attn
    # out_proj. The uniform-attention names the other six drivers use do not exist
    # here, so this list cannot be shared with them.
    "structural": [
        # The last layer on its own. Layer 31 is a full-attention layer, so the
        # single cell is available at the top of the stack rather than at the last
        # full-attn layer below it.
        (f"full-attn · layer {N_LAYERS - 1} · o_proj", "cosine"),
        ("all layers · all projections", "cosine"),
        ("full-attn · late third", "cosine"),
        ("full-attn · q_proj (whole)", "cosine"),
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
        surrogates=False,
        select={k: v for k, v in SELECT.items() if k in levels},
        crosslevel_only=True,
        n_expected=N_EXPECTED,
        datasets=DATASETS,
        embedder=EMBEDDER,
        dataset_embedder=DATASET_EMBEDDER,
        source="figures/simplex3_dolly_qwen/make_figures.py",
    )


if __name__ == "__main__":
    main()
