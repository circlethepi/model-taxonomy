#!/usr/bin/env python
"""Regenerate the whole visualization suite for the simplex3_qwen experiment.

The qwen driver. Everything this used to do now lives in
:mod:`src.plots.simplex_suite` — read that module's docstring for what the suite
draws, why some cells are structurally absent, and how ``06_pairwise`` reuse
works. What stays here is the qwen run's identity and the command line, which is
unchanged: the same flags, the same defaults, the same figures.

Usage
-----
    python scripts/make_simplex3_figures.py                  # everything
    python scripts/make_simplex3_figures.py --level functional
    python scripts/make_simplex3_figures.py --skip-sweep     # omit the slow one
    python scripts/make_simplex3_figures.py --with-surrogate # add the fleet transforms
    python scripts/make_simplex3_figures.py --cache-root PATH  # explicit cache
    python scripts/make_simplex3_figures.py --no-cache          # force a cold run

This driver draws the **full** grid. For the same run restricted to the
perspectives that carry over to the llama3i comparison, see
``figures/simplex3_qwen_v4/make_figures.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plots import simplex_suite as suite  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--level", action="append",
                    choices=["behavioral", "functional", "structural",
                             "dataset_embedding"],
                    help="restrict to one or more levels (default: all)")
    # A new directory rather than overwriting `figures/simplex3_qwen/`: those are
    # the row-permuted figures (docs/notes/row_order_bug.md), and keeping them
    # side by side is what lets the correction be checked rather than taken on
    # trust. The PNGs are gitignored here as they are there; only the agreement
    # table is tracked.
    ap.add_argument("--outdir",
                    default=str(REPO_ROOT / "figures" / "simplex3_qwen_v2"))
    ap.add_argument("--skip-sweep", action="store_true",
                    help="skip the 33-layer functional sweep")
    ap.add_argument("--skip-detail", action="store_true",
                    help="skip the per-metric detail figures")
    # Dropped from the default grid on the evidence in
    # figures/simplex3_qwen_v3/crosslevel_scores.csv: no centered or whitened
    # perspective wins at any level. They are dominated rather than broken —
    # functional centered reaches 0.955 against a 0.975 best — except behavioral
    # CKA, where both go negative (-0.548 centered, -0.674 whitened). The code
    # path survives behind the flag so the finding can be re-checked without
    # reinstating code. When run, these are computed uncached: G1 sends them to
    # the plain `_distances`, so they touch neither 06_pairwise nor
    # 07_collections.
    ap.add_argument("--with-surrogate", action="store_true",
                    help="also compute the centered/whitened fleet transforms, "
                         "which are off by default")
    ap.add_argument("--cache-root", default=None,
                    help="the shared cache to read models from and to reuse "
                         f"distance matrices in (default: {suite.CACHE_ROOT})")
    # A warm run reproducing a cold run exactly is the only real test that the
    # reuse is correct, and it cannot be run without a supported way to force the
    # cold one. Compare the `matrix_sha256` column of the two runs'
    # `crosslevel_scores.csv`.
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute everything, ignoring stored results "
                         "(still writes them back)")
    # Run identity.  Architecture is derived from the checkpoint; what cannot be
    # derived is *which run* to plot, so that comes from here.
    ap.add_argument("--base-model", default=suite.BASE_MODEL,
                    help=f"the suite's base model (default: {suite.BASE_MODEL})")
    ap.add_argument("--draw-recipe-hash", default=suite.DRAW["recipe_hash"])
    ap.add_argument("--draw-n", type=int, default=suite.DRAW["n_samples"])
    ap.add_argument("--draw-seed", type=int, default=suite.DRAW["seed"])
    ap.add_argument("--draw-format-id", default=suite.DRAW["prompt_format_id"],
                    help="prompt_format_id of the query draw; '' for a raw suite")
    # Which corpus. 03_adapters/<base_slug> holds every adapter for a base model
    # whatever it was trained on -- that is what makes the cache shared -- so
    # once a second dataset is trained on this model an unfiltered scan returns
    # both and the count guard trips. Repeatable, so a deliberately
    # cross-dataset comparison stays expressible.
    ap.add_argument("--dataset", action="append", dest="datasets",
                    help="restrict to a corpus, by recipe-name prefix (yahoo, "
                         "dolly, oasst1) or dataset_id; repeat for several. "
                         "Default: every corpus in the cache.")
    args = ap.parse_args()

    draw = {"recipe_hash": args.draw_recipe_hash, "n_samples": args.draw_n,
            "seed": args.draw_seed}
    if args.draw_format_id:
        draw["prompt_format_id"] = args.draw_format_id

    suite.run_suite(
        base_model=args.base_model,
        draw=draw,
        outdir=args.outdir,
        cache_root=args.cache_root,
        levels=args.level,
        skip_sweep=args.skip_sweep,
        skip_detail=args.skip_detail,
        surrogates=args.with_surrogate,
        no_cache=args.no_cache,
        datasets=args.datasets,
        source="scripts/make_simplex3_figures.py",
    )


if __name__ == "__main__":
    main()
