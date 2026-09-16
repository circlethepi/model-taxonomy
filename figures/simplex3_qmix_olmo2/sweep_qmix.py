#!/usr/bin/env python
"""Score every qmix point against the simplex the adapters were trained on.

**qmix** (*query mixture*) -- the axis of query-set *composition*: what fraction
of the 100-row probe is the corpus the adapters were trained on, the rest being a
diluting corpus they never saw.  One value of qmix is one ``(yahoo percentage,
diluent)`` pair.  See ``docs/terminology.md`` and
``docs/notes/qmix_dataset_composition.md``.

The question: the ground truth never moves -- these are the same sixteen
adapters, trained on the same sixteen mixtures, every time -- so how far does
each taxonomy level's *recovery* of that fixed simplex degrade as the probe it is
read through is diluted?

**What is fixed and what varies is the reverse of every other sweep here.**  The
nsweep next door varies the training draw and holds the probe; the group-size
sweep varies how many adapters are in a collection and holds both.  Here the
adapters are one unchanging fleet and the *instrument* moves.  That is why there
is no realized-versus-requested truth decomposition in this file: largest-
remainder allocation can misspecify a training mixture, and none of these
adapters' training mixtures are being touched.

**Only the query-dependent perspectives are scored.**  Four of the seven
canonical rows -- the three structural ones and the dataset-embedding one -- do
not read the query set at all: structural reads LoRA weights and dataset reads
the training draw.  Under this axis they are flat by construction, and plotting a
flat line next to four that move invites reading it as a result.  They are
already measured, at full strength, in the simplex3_olmo2 figures.

**The 100% point is not the canonical probe.**  The yahoo component of a qmix
draw is unstratified -- all ten topics as one pool -- where the canonical probe
draws the even g1/g2/g3 mixture.  So the undiluted point here is a new draw with
its own recipe hash and its own cached generations.  Read this curve against
itself, across yahoo percentage; do not read its 100% point against the
behavioral row of figure 2.

Usage::

    python figures/simplex3_qmix_olmo2/sweep_qmix.py
    python figures/simplex3_qmix_olmo2/sweep_qmix.py --perspective behavioral
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.experiments.query_mixture_spec import QMIX_SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import raw_mixture_pcts, sort_by_mixture  # noqa: E402

# The canonical perspectives are defined once, next to the group-size sweep, and
# imported rather than copied -- they are the standing per-level defaults for
# this project, not a property of any one experiment, and two divergent copies
# would be a silent way for two figures to stop being comparable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simplex_collection_size"))
from sweep_group_size import _META_KEYS, canonical_perspectives  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = Path("/weka/scratch/jhu/cpriebe1/MO/model-taxonomy")
BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"
CACHE_ROOT = REPO / "results" / "shared_cache"
TREE = "simplex3_qmix_olmo2"

#: The perspectives whose surrogate is read *through the query set*.  The others
#: are query-independent; see the module docstring.
QUERY_DEPENDENT = ("behavioral", "behavioral_greedy",
                   "functional_all", "functional_last")

#: Which selector key each perspective's draw lives under.  The three selectors
#: are named differently per level and there is no common accessor, so the map is
#: written down rather than guessed at from the taxonomy name.
SELECTOR_KEY = {"behavioral": "behavioral_selector",
                "functional": "functional_selector"}

#: One row per (perspective, arm, yahoo percentage, draw seed).
FIELDS = ["perspective", "taxonomy", "metric", "arm", "yahoo_pct", "seed",
          "recipe_hash", "n_models", "dcor", "disparity"]


def query_draws(exp_dir: Path, results_dir: Path, qmix) -> list[dict]:
    """Every qmix query draw, read from the emitted tree rather than rederived.

    The draw key is ``{recipe_hash, n_samples, seed, prompt_format_id}`` and each
    of those four already exists somewhere authoritative: the hash in the recipe
    the build step wrote, the format id in the config's own ``prompt_format``,
    and n and seed in the config's dataset block.  Recomputing any of them here
    would be a second definition that can disagree with the first -- and a draw
    key that disagrees by one field does not raise, it silently finds nothing.
    """
    # `_utils.load_recipe` is the one loader that dispatches on `recipe_type`;
    # re-reading the JSON here would be a second definition of that dispatch.
    sys.path.insert(0, str(REPO / "scripts"))
    from _utils import load_recipe  # noqa: E402
    from src.datasets._chat_projection import PromptFormat

    draws = []
    for arm, pct in qmix.points():
        name = qmix.recipe_name(arm, pct)
        recipe_path = results_dir / "datasets" / f"{name}.recipe.json"
        if not recipe_path.exists():
            raise SystemExit(
                f"no recipe at {recipe_path}. The build step writes it; run one "
                f"qmix job (or `run_experiment.py ... --steps build`) first."
            )
        rhash = load_recipe(recipe_path).recipe_hash()
        for seed in qmix.seeds:
            cfg_path = exp_dir / f"behavioral_{name}_s{seed:02d}.yaml"
            cfg = yaml.safe_load(cfg_path.read_text())
            fmt_id = PromptFormat.from_config(cfg.get("prompt_format")).format_id()
            draws.append({
                "arm": "none" if arm is None else arm.key,
                "yahoo_pct": pct,
                "seed": seed,
                "draw": {"recipe_hash": rhash,
                         "n_samples": cfg["extraction"]["n_queries"],
                         "seed": seed,
                         **({} if fmt_id is None
                            else {"prompt_format_id": fmt_id})},
            })
    return draws


def with_draw(spec: dict, draw: dict) -> dict:
    """*spec* with its level selector pointed at *draw*.

    Every other field of the selector -- pooling, view, normalisation, the
    embedder, the sampling hash -- is left exactly as the canonical perspective
    defines it.  The draw is the one thing this experiment varies, and the whole
    point of importing the perspectives is that nothing else does.
    """
    key = SELECTOR_KEY[spec["taxonomy"]]
    return {**spec, key: {**spec[key], "draw": draw}}


def point_matrix(index, ids, cache_root, name, spec, *, use_cache=True):
    """The one distance matrix for *name* over the 16 models at one qmix point."""
    kwargs = {k: v for k, v in spec.items() if k not in _META_KEYS}
    dm, _ = build_taxonomy_artifacts(
        index, spec["taxonomy"], spec["metric"], cache_root=cache_root,
        n_components=(2,), use_cache=use_cache, id_scheme="model_id",
        label="qmix point", **kwargs,
    )
    return dm.reindex(list(ids))


def score_point(dm, ids, vertex_names):
    """dCor* and Procrustes disparity against the one fixed truth.

    One truth, not two.  The requested/realized split the nsweep carries is about
    training-mixture misspecification at small draws, and nothing here varies a
    training draw -- these sixteen adapters are the ``n=1000`` fleet throughout,
    where the requested and realized mixtures already coincide.
    """
    weights = suite.truth_weights(ids)
    tdm = simplex_distance_matrix(weights, list(ids), vertex_names)
    tgeo = simplex_geometry(weights, list(ids), vertex_names)
    return (
        dcor_vs_truth(dm, tdm),
        disparity_vs_truth(dm, tgeo, random_state=suite.MDS_SEED,
                           n_components=tgeo.coordinates.shape[1]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--cache-root", default=str(CACHE_ROOT))
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--tree", default=TREE)
    ap.add_argument("--outdir", default=str(HERE))
    ap.add_argument("--perspective", action="append", dest="perspectives",
                    help="repeatable; default is every query-dependent perspective")
    ap.add_argument("--spec", default="yahoo",
                    help="DataSimplexSpec key naming the fixed adapter fleet")
    ap.add_argument("--qmix-spec", default="yahoo", help="QMixSpec key")
    ap.add_argument("--n-expected", type=int, default=16,
                    help="models per point; any other count is fatal")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit-points", type=int, default=None,
                    help="score only the first N query sets (a timing probe)")
    ap.add_argument("--out", default="qmix_scores.csv")
    args = ap.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve()
    if not cache_root.exists():
        raise SystemExit(f"no cache at {cache_root}")
    repo = Path(args.repo)

    # Bind the architecture before reading the perspectives: two of them name
    # "the last layer", which is a position in a stack rather than a number.
    suite.apply_architecture(suite.architecture(args.base_model))
    print(f"model: {args.base_model}  ({suite.N_LAYERS} layers, "
          f"{suite.N_STATES} states)", flush=True)

    qmix = QMIX_SPECS[args.qmix_spec]
    draws = query_draws(repo / "experiments" / args.tree,
                        repo / "results" / args.tree, qmix)
    if args.limit_points:
        draws = draws[:args.limit_points]
    print(f"{len(draws)} query sets over {len(qmix.points())} compositions "
          f"x {len(qmix.seeds)} draw seeds", flush=True)

    spec = SPECS[args.spec]
    wanted = {tuple(m) for m in spec.mixture_pcts()}

    # No draw filter on the scan.  The `behavioral_repr` / `functional_repr`
    # tokens report on ONE draw, and this experiment has 150 -- so a filter here
    # could only be right for one point.  The mixture filter still matters and is
    # not optional: the 1004-adapter group-size pool is also yahoo, also this
    # base model and also (n_samples=1000, seed=0), so without it the scan
    # returns 1015 models and the completeness guard below fires on every point.
    index = scan_cache(cache_root, base_model_id=args.base_model,
                       datasets=["yahoo"])
    index = CacheIndex([e for e in index.entries
                        if raw_mixture_pcts(e.model_id) in wanted],
                       index.cache_root)
    ids = sort_by_mixture(list(index.model_ids))
    if len(ids) != args.n_expected:
        raise SystemExit(
            f"found {len(ids)} adapters, expected {args.n_expected}. The ground "
            f"truth is only unambiguous at the full simplex, so this is fatal "
            f"rather than skippable. Ids: {ids[:20]}")
    vertex_names = [f"g{j + 1}"
                    for j in range(suite.truth_weights(ids).shape[1])]

    specs = canonical_perspectives()
    unknown = set(args.perspectives or ()) - set(specs)
    if unknown:
        raise SystemExit(f"unknown perspective(s) {sorted(unknown)}. "
                         f"Choose from {sorted(specs)}")
    names = args.perspectives or list(QUERY_DEPENDENT)
    off_axis = [n for n in names if n not in QUERY_DEPENDENT]
    if off_axis:
        print(f"note: {off_axis} do not read the query set; their curves are "
              f"flat by construction under this axis.", flush=True)

    rows = []
    t_start = time.time()
    for name in names:
        base_spec = specs[name]
        for d in draws:
            s = with_draw(base_spec, d["draw"]) if name in QUERY_DEPENDENT \
                else dict(base_spec)
            t0 = time.time()
            dm = point_matrix(index, ids, cache_root, name, s,
                              use_cache=not args.no_cache)
            dcor, disp = score_point(dm, ids, vertex_names)
            rows.append({
                "perspective": name, "taxonomy": s["taxonomy"],
                "metric": s["metric"], "arm": d["arm"],
                "yahoo_pct": d["yahoo_pct"], "seed": d["seed"],
                "recipe_hash": d["draw"]["recipe_hash"], "n_models": len(ids),
                "dcor": dcor, "disparity": disp,
            })
            print(f"  {name:<20} {d['arm']:<9} yahoo={d['yahoo_pct']:<4} "
                  f"s={d['seed']:<3} dcor={dcor:.4f} disp={disp:.4f} "
                  f"{time.time() - t0:6.1f}s", flush=True)

    out = Path(args.outdir) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {out} in "
          f"{time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
