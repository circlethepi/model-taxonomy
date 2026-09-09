#!/usr/bin/env python
"""The collection-size suite: how many models a taxonomy score needs.

Every other figure in this project scores a **16-model** collection.  This one
holds the taxonomy fixed and moves the collection size instead, so that a score
can be read as "this level recovers the simplex" rather than "this level
recovers the simplex *at sixteen models*".

The pool
--------
999 yahoo mixtures drawn without replacement from the non-vertex points of the
one-percent grid over three groups, each fine-tuned on OLMo-2-0425-1B-Instruct
with the same recipe (1000 rows, seed 0, LoRA rank 16), plus the three vertices
and the 33/33/33 centroid as **reference models** -- 1003 in all.

Two filters make that pool, and neither is optional:

* **The training recipe.**  ``03_adapters/allenai--OLMo-2-0425-1B-Instruct``
  holds every adapter ever trained on this checkpoint, and the dataset-size
  sweep trains other row counts there.  ``--train-n 1000 --train-seed 0`` keeps
  this pool's own recipe.
* **The simplex3 grid, by name.**  ``figures/simplex3_olmo2`` trains sixteen
  quarter-step mixtures on the same checkpoint, the same corpus and the same
  recipe, so no scan argument separates them from the pool.  Four of the sixteen
  *are* this suite's reference models and are kept; :data:`SIMPLEX3_EXTRA` names
  the other twelve and they are dropped, because a pool that mixes 999 uniform
  draws with twelve grid points is no longer a uniform sample of the simplex.

What is scored
--------------
The seven standing perspectives, in ``sweep_group_size.canonical_perspectives``:
the dataset level's mean embedding under euclidean, the structural level at
three scopes, the functional level at two, and the behavioral level's per-query
replicate mean -- each against the simplex ground truth under dCor* and
Procrustes disparity, and each under both reference variants.

Outputs, all in this directory
------------------------------
``group_size_scores.csv``   one row per (perspective, n, replicate)
``group_members.csv``       which models each replicate drew
``run_config.json``         pool, grid, seeds and perspectives of the run
``fig_group_size.png``      the figure
``group_size_summary.md``   the same medians as a table

Usage
-----
::

    python figures/simplex_collection_size/make_figures.py
    python figures/simplex_collection_size/make_figures.py --plot-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Pinned before numpy loads its BLAS -- see the note in src/plots/simplex_suite.py.
os.environ.setdefault("MODEL_TAXONOMY_THREADS", "1")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]

BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: The canonical checkout's cache.  A worktree has no ``results/`` of its own and
#: ``suite.CACHE_ROOT`` is derived from the module's location, so a run from one
#: would otherwise resolve to a directory that does not exist.
CACHE_ROOT = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache")

#: The twelve ``figures/simplex3_olmo2`` mixtures that are **not** this pool's
#: reference models: the quarter-step grid minus the three vertices and the
#: centroid.  Named rather than derived so that adding a mixture to that suite
#: cannot silently add one here.
SIMPLEX3_EXTRA = [
    "yahoo_000g1_025g2_075g3", "yahoo_000g1_050g2_050g3",
    "yahoo_000g1_075g2_025g3", "yahoo_025g1_000g2_075g3",
    "yahoo_025g1_025g2_050g3", "yahoo_025g1_050g2_025g3",
    "yahoo_025g1_075g2_000g3", "yahoo_050g1_000g2_050g3",
    "yahoo_050g1_025g2_025g3", "yahoo_050g1_050g2_000g3",
    "yahoo_075g1_000g2_025g3", "yahoo_075g1_025g2_000g3",
]

#: Group sizes.  Roughly geometric, because the ``1/(n(n-3))`` factor in dCor*
#: and the similarity-transform degeneracy in Procrustes both live on a ratio
#: scale.  It stops at 500 rather than 1000: the sampled pool is 999 models, so
#: 1000 is not available at all, and 999 would be a single group -- one
#: measurement with no spread, drawn a hundred times.  To add it later::
#:
#:     python figures/simplex_collection_size/sweep_group_size.py \\
#:         --n-grid 999 --append
#:
#: which keeps every row already on disk and appends that size alone.
N_GRID = "5,10,20,50,100,200,500"

REPLICATES = 100

#: Seed for the permute-and-partition shuffles.  Reproduces the groups only
#: against an identical pool in an identical order, which is why
#: ``group_members.csv`` records the membership itself as well.
SEED = 0


def run(script: str, extra: list[str]) -> None:
    cmd = [sys.executable, str(HERE / script), *extra]
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def perspective_names() -> list[str]:
    """The perspectives the sweep would score, without importing its whole module."""
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(REPO_ROOT))
    from sweep_group_size import canonical_perspectives  # noqa: E402
    from src.plots import simplex_suite as suite  # noqa: E402

    suite.apply_architecture(suite.architecture(BASE_MODEL))
    return list(canonical_perspectives())


def fan_out(sweep_args: list[str], jobs: int, tmp: Path) -> None:
    """Score one perspective per process, then merge into one CSV.

    Worth having rather than doing by hand, because the cost here is **MDS, and
    MDS does not parallelise inside one process**.  At ``n=500`` the pool yields
    a single disjoint group per shuffle, so the membership memo never hits and
    every replicate is its own fit: ~1400 fits at 500 points for the whole
    suite, hours of one core.  Perspectives are completely independent -- they
    share only the pool matrices, which are read from ``07_collections`` -- so
    splitting on them is exact rather than approximate.

    The scores are those of a serial run -- the groups are a function of the pool
    and the shuffle seed, not of which process drew them -- and only the row
    *order* differs, since the merge concatenates one perspective at a time
    where a serial run interleaves them.  Nothing downstream reads row order.
    What the merge has to repair is ``run_config.json``, which each process
    writes listing only the one perspective it was given.
    """
    tmp.mkdir(parents=True, exist_ok=True)
    names = perspective_names()
    running: list[tuple[str, subprocess.Popen]] = []

    def reap(limit: int) -> None:
        while len(running) >= limit:
            for i, (nm, p) in enumerate(list(running)):
                if p.poll() is not None:
                    running.pop(i)
                    if p.returncode != 0:
                        raise SystemExit(
                            f"{nm} failed with exit {p.returncode}; see "
                            f"{tmp / nm}.log")
                    print(f"  done: {nm}", flush=True)
                    break
            else:
                time.sleep(2)

    for nm in names:
        reap(jobs)
        out = tmp / nm
        out.mkdir(parents=True, exist_ok=True)
        log = (tmp / f"{nm}.log").open("w")
        cmd = [sys.executable, str(HERE / "sweep_group_size.py"),
               *sweep_args, "--perspective", nm]
        cmd[cmd.index("--outdir") + 1] = str(out)
        print(f"  start: {nm}", flush=True)
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                             cwd=REPO_ROOT)
        running.append((nm, p))
    reap(1)

    scores, config = [], None
    for nm in names:
        with (tmp / nm / "group_size_scores.csv").open() as fh:
            rows = fh.read().splitlines()
        scores.append((rows[0], rows[1:]))
        cfg = json.loads((tmp / nm / "run_config.json").read_text())
        if config is None:
            config = cfg
        else:
            config["perspectives"].update(cfg["perspectives"])
    header = scores[0][0]
    body = [line for _, lines in scores for line in lines]
    (HERE / "group_size_scores.csv").write_text(
        "\n".join([header, *body]) + "\n")
    # Membership does not depend on the perspective, so any process's copy is
    # the whole record; taking the first is not a shortcut.
    shutil.copy(tmp / names[0] / "group_members.csv", HERE / "group_members.csv")
    (HERE / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"merged {len(body)} row(s) from {len(names)} perspective(s)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.split("Usage")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plot-only", action="store_true",
                    help="redraw from the existing group_size_scores.csv")
    ap.add_argument("--sweep-only", action="store_true",
                    help="score without redrawing")
    ap.add_argument("--n-grid", default=N_GRID)
    ap.add_argument("--replicates", type=int, default=REPLICATES)
    ap.add_argument("--check-matrices", action="store_true",
                    help="build the pool matrices, report whether they are "
                         "complete, symmetric and zero-diagonal, and stop")
    ap.add_argument("--jobs", type=int, default=1,
                    help="score this many perspectives at once, in separate "
                         "processes, then merge. The output is identical to a "
                         "serial run; MDS is the cost and does not thread")
    ap.add_argument("--tmp", default=None,
                    help="scratch directory for --jobs (default: <outdir>/.fanout)")
    args, rest = ap.parse_known_args()

    sweep_args = [
        "--base-model", BASE_MODEL,
        "--cache-root", str(CACHE_ROOT),
        "--outdir", str(HERE),
        "--dataset", "yahoo",
        "--train-n", "1000",
        "--train-seed", "0",
        "--n-grid", args.n_grid,
        "--replicates", str(args.replicates),
        "--seed", str(SEED),
    ]
    for m in SIMPLEX3_EXTRA:
        sweep_args += ["--exclude-mixture", m]
    if args.check_matrices:
        sweep_args.append("--check-matrices")

    if not args.plot_only:
        if args.jobs > 1 and not args.check_matrices:
            fan_out(sweep_args + rest, args.jobs,
                    Path(args.tmp) if args.tmp else HERE / ".fanout")
        else:
            run("sweep_group_size.py", sweep_args + rest)
    if not (args.sweep_only or args.check_matrices):
        run("plot_group_size.py", ["--outdir", str(HERE)])


if __name__ == "__main__":
    main()
