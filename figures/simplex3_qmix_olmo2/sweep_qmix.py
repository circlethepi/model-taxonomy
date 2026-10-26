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

import numpy as np  # noqa: E402

from src.analysis.bridge import fit_geometry  # noqa: E402
from src.analysis.comparison import build_taxonomy_artifacts  # noqa: E402
from src.analysis.configurations import procrustes_compare  # noqa: E402
from src.analysis.discovery import CacheIndex, scan_cache  # noqa: E402
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.cache.activation_cache import ActivationCache  # noqa: E402
from src.cache.generated_text_cache import GeneratedTextCache  # noqa: E402
from src.experiments.data_simplex_spec import SPECS  # noqa: E402
from src.experiments.query_mixture_spec import QMIX_SPECS  # noqa: E402
from src.plots import simplex_suite as suite  # noqa: E402
from src.plots.simplex import (  # noqa: E402
    mixture_label, mixture_weights, raw_mixture_pcts, sort_by_mixture,
)

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

#: One row per (perspective, arm, yahoo percentage, kind, seed, model): 2-D MDS
#: coordinates.  Every alignment happens here rather than at plot time, so the
#: figure driver reads coordinates and draws them -- see `point_geometry` for
#: what each `kind` is and why the three are not interchangeable.
GEOMETRY_FIELDS = ["perspective", "taxonomy", "metric", "arm", "yahoo_pct",
                   "kind", "seed", "model_id", "mixture", "dim1", "dim2"]

#: `seed` of a row that summarises the seeds rather than being one of them.
#: Not a seed; a sentinel, so the column stays integer-typed.
POOLED = -1

#: The stage cache each taxonomy's surrogate is read out of.  Used only to ask
#: whether a draw exists at all; see `draw_available`.
STAGE_CACHE = {"functional": ActivationCache, "behavioral": GeneratedTextCache}


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


#: The behavioral embedder this tree generated under: multilingual, because the
#: oasst1-zh arm's probes are Chinese and so are the generations they elicit.
#: ``simplex_suite.EMBEDDER`` is yahoo's English-only v1.5 and would find nothing
#: here -- the hash is a path component, so a mismatch reads as an empty cache
#: rather than as an error. The value is the one the dolly and oasst1 trees
#: already use; see ``simplex_suite.EMBEDDER``'s own comment.
QMIX_EMBEDDER = "0b579825f703fb21"   # nomic-embed-text-v2-moe, search_document


def with_draw(spec: dict, draw: dict) -> dict:
    """*spec* with its level selector pointed at *draw*, on this tree's embedder.

    Every other field of the selector -- pooling, view, normalisation, the
    sampling hash -- is left exactly as the canonical perspective defines it.
    The draw is the one thing this experiment varies, and the whole point of
    importing the perspectives is that nothing else does.

    The embedder is the one exception, and it is not a free choice: it is a
    property of what was *written*, not of how it is read. These generations were
    embedded with v2-moe, so asking for v1.5 would address a cache entry that
    does not exist.
    """
    key = SELECTOR_KEY[spec["taxonomy"]]
    sel = {**spec[key], "draw": draw}
    if "embedder_hash" in sel:
        sel["embedder_hash"] = QMIX_EMBEDDER
    return {**spec, key: sel}


def draw_available(taxonomy, cache_root, base_model, model_ids, draw) -> bool:
    """True when *every* model of the fleet has *draw* in the stage it is read from.

    A qmix tree is emitted all at once and submitted as the cluster allows, so a
    scoring run routinely meets a tree that is only partly extracted: the ten
    draw seeds of one composition are ten independent jobs.  This is the cheap
    honest test for "has this seed run yet" -- a draw whose extraction never
    happened has no directory in ``04_activations`` / ``05_generated`` at all.

    Asking the filesystem beats asking the loader.  ``build_taxonomy_artifacts``
    resolves every model's representation *before* it consults the collection
    cache, so a missing draw surfaces as a ``ValueError`` several hundred adapter
    loads in, with sixteen models' work already thrown away.

    **All-or-nothing on purpose.**  A partly-extracted draw would give a distance
    matrix over a *subset* of the simplex, and the ground truth is only
    unambiguous at the full sixteen points -- the same reason the completeness
    guard in `main` is fatal rather than skippable.
    """
    cache = STAGE_CACHE[taxonomy](cache_root)
    return all(cache.draw_dir(base_model, m, draw).exists() for m in model_ids)


def mean_distance_matrix(per_seed, order):
    """The **pooled** matrix: the per-seed matrices averaged cell by cell.

    *per_seed* maps draw seed to that seed's :class:`DistanceMatrix`; *order* is
    the mixture-sorted id list the average is labelled by.  Every matrix is
    reindexed onto that order first, so the mean is of the same cell across seeds
    rather than of whatever row order each fit came back in.

    Unlike the init-seed sweep next door, the sixteen *models* here are literally
    the same sixteen adapters at every seed -- only the probe they were measured
    through moved.  So this is an average over *measurements* of one fleet, and a
    single seed is a legitimate degenerate case of it: at ``len(per_seed) == 1``
    the pooled matrix is that seed's own, which is exactly what it should be.
    """
    from src.core.distance import DistanceMatrix

    ref = per_seed[min(per_seed)]
    stack = []
    for seed in sorted(per_seed):
        dm = per_seed[seed]
        missing = [m for m in order if m not in set(dm.model_ids)]
        if missing:
            raise SystemExit(
                f"seed {seed} is missing model(s) {missing}; the seeds must "
                "cover the same simplex for a mean over them to mean anything.")
        stack.append(np.asarray(dm.reindex(list(order)).matrix, dtype=np.float64))
    return DistanceMatrix(matrix=np.mean(stack, axis=0), model_ids=list(order),
                          metric=ref.metric, taxonomy=ref.taxonomy)


def _vertex_label(labels, group):
    """The label of the adapter trained on *group* alone, e.g. 100/0/0.

    Found by weight rather than by spelling, so the ``NNNgK`` field widths and
    the ``_nNNNN_sNN_rNN_iNN_bNNNN`` tail are not this function's problem.
    """
    for label in labels:
        weights = mixture_weights(label)
        if group < len(weights) and weights[group] > 0.99:
            return label
    raise ValueError(f"no pure group-{group + 1} vertex among {len(labels)} "
                     "labels, so the canonical orientation is undefined")


def canonical_frame(geo):
    """The centring and the 2x2 map that put a configuration in the house frame.

    **Why this is not `src.plots.simplex.align_to_simplex`.**  That function does
    the same three things -- centre, rotate pure g1 onto +y, reflect pure g2 into
    +x -- but it returns *transformed coordinates* for one configuration rather
    than a frame that can be applied to several.  An overlay needs the second:
    the pooled fit, the mean and each per-seed fit have already been
    Procrustes-superimposed on one another, and re-aligning each one
    independently would undo exactly that superposition and leave a panel whose
    spread is arbitrary orientation again.  Hence the (derive, apply) split.

    ``figures/fig_structural_sweep/sweep_initsweep.py`` carries a near-identical
    pair for the init-seed overlay.  They are kept separate rather than one
    importing the other: a cross-experiment import of two small helpers out of a
    1000-line driver about a different sweep is a fragile coupling, and the
    non-fragile fix is to move the pair into ``src/plots/simplex.py`` beside
    ``align_to_simplex``, which is a shared-module change neither experiment
    should make on its own.  If a third caller appears, that is the move.

    MDS fixes coordinates only up to rotation, reflection and scale, so two
    panels of the same simplex can be mirror images of each other and nothing is
    wrong.  That is fine for one picture and bad for a grid of them: a reader
    comparing eight dilution levels along a row would have to re-derive which way
    is which in every panel.  This pins the remaining freedom by the *content*
    rather than by the fit -- the pure-g1 vertex is rotated onto the positive y
    axis, and the pure-g2 vertex is reflected into positive x if it is not there
    already.  Every panel then shows the simplex the same way up.

    Returns ``(centroid, M)``; apply with :func:`orient`.  Scale is deliberately
    untouched, so this composes with a Procrustes superposition without undoing
    its fitted scale.
    """
    coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
    labels = list(geo.model_ids)
    centroid = coords.mean(axis=0)
    at = {lab: coords[i] - centroid for i, lab in enumerate(labels)}

    x, y = at[_vertex_label(labels, 0)]
    # Rotate by the angle that carries this vertex from where it is to due north.
    alpha = np.pi / 2 - np.arctan2(y, x)
    c, sn = np.cos(alpha), np.sin(alpha)
    M = np.array([[c, -sn], [sn, c]])
    if (M @ at[_vertex_label(labels, 1)])[0] < 0:
        # Mirror in the y axis. The g1 vertex is on that axis by construction and
        # so is fixed; only the handedness of the panel changes.
        M = np.array([[-1.0, 0.0], [0.0, 1.0]]) @ M
    return centroid, M


def orient(geo, frame):
    """Apply a :func:`canonical_frame` to one geometry, keeping its labels."""
    from src.core.geometry import GeometryResult

    centroid, M = frame
    coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
    moved = (coords - centroid) @ M.T
    return GeometryResult(
        coordinates=moved.astype(np.float32), model_ids=list(geo.model_ids),
        method=geo.method, taxonomy=geo.taxonomy, n_components=2,
        stress=geo.stress, metadata={**(geo.metadata or {}),
                                     "oriented": "g1 vertex north, g2 vertex east"},
    )


def mean_of(geos):
    """Per-model centroid over geometries that already share one frame.

    **The mean location**: where the aligned per-seed fits put each adapter, on
    average.  This is the overlay's own centre, and it is a third object,
    distinct from both the pooled fit and any one seed: the pooled fit averages
    *distances* and embeds afterwards, while this averages *coordinates* after
    reconciling frames.  The two disagree because MDS is not linear, which is
    why the figure carries both rather than choosing.
    """
    from collections import defaultdict

    from src.core.geometry import GeometryResult

    by_id = defaultdict(list)
    ref = geos[0]
    for geo in geos:
        coords = np.asarray(geo.coordinates, dtype=np.float64)[:, :2]
        for i, label in enumerate(geo.model_ids):
            by_id[label].append(coords[i])
    keys = list(ref.model_ids)
    means = np.vstack([np.mean(by_id[k], axis=0) for k in keys])
    return GeometryResult(
        coordinates=means.astype(np.float32), model_ids=keys, method=ref.method,
        taxonomy=ref.taxonomy, n_components=2, stress=0.0,
        metadata={"derived": "mean over cross-seed-aligned embeddings"},
    )


def point_geometry(per_seed_dm, order):
    """The three configurations one qmix point contributes to the MDS figure.

    Returns ``{kind: GeometryResult}`` with every kind in one frame, turned the
    same way up as every other panel by :func:`canonical_frame`:

    ``pooled``
        The pooled matrix embedded -- distances averaged over seeds, then one MDS
        fit.  The panel's backdrop.

    ``mean``
        The **mean location** per adapter: each model's position averaged over
        the *aligned* per-seed fits.  See :func:`mean_of` for why this is not the
        same point as the pooled fit's.

    ``seed_aligned``
        Each seed's own 16-point fit, Procrustes-superimposed on the pooled fit.
        This is the overplotted layer, and the superposition is not cosmetic:
        MDS fixes coordinates only up to rotation, reflection and scale, so the
        raw fits overlaid would show a spread that is mostly arbitrary
        orientation.  What survives alignment is a genuine disagreement between
        draws of the probe.

    Also returns ``reference``: the pooled fit *in the frame the seeds were
    aligned into*.  ``procrustes_compare`` scales both configurations to unit
    Frobenius norm, so the raw ``pooled`` is off this scale by a factor of the
    surrogate's own distances -- which run from ~1 for structural to ~1e-3 for
    functional.  Drawing ``seed_aligned`` against raw ``pooled`` would therefore
    put the cloud and its backdrop at two different sizes; ``reference`` is the
    one to draw them against.
    """
    seeds = sorted(per_seed_dm)
    pooled_dm = mean_distance_matrix(per_seed_dm, order)
    pooled_geo = fit_geometry(pooled_dm, method="mds", n_components=2,
                              random_state=suite.MDS_SEED)
    per_seed_geo = {
        s: fit_geometry(per_seed_dm[s].reindex(list(order)), method="mds",
                        n_components=2, random_state=suite.MDS_SEED)
        for s in seeds
    }

    aligned, reference = {}, None
    for s in seeds:
        fit = procrustes_compare(pooled_geo, per_seed_geo[s])
        aligned[s] = fit.aligned_b
        if reference is None:
            # Every seed is aligned to the same reference, so which seed produced
            # this copy of it does not matter.
            reference = fit.aligned_a

    mean_geo = mean_of([aligned[s] for s in seeds])
    # One frame for the whole panel -- cloud, mean and reference together.
    frame = canonical_frame(mean_geo)
    out = {"pooled": orient(reference, frame), "mean": orient(mean_geo, frame)}
    return out, {s: orient(aligned[s], frame) for s in seeds}, pooled_dm


def geometry_rows(geo, name, spec, d_arm, pct, kind, seed):
    """Coordinate rows for one oriented configuration."""
    coords = np.asarray(geo.coordinates, dtype=np.float64)
    return [{
        "perspective": name, "taxonomy": spec["taxonomy"],
        "metric": spec["metric"], "arm": d_arm, "yahoo_pct": pct,
        "kind": kind, "seed": seed, "model_id": m,
        "mixture": mixture_label(m),
        "dim1": float(coords[i, 0]), "dim2": float(coords[i, 1]),
    } for i, m in enumerate(geo.model_ids)]


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


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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
    # The training recipe the fixed fleet shares.  Defaults, not discoveries:
    # this experiment is *about* holding the adapters still, so the four fields
    # that identify them are pinned by name rather than inferred from whatever
    # the cache happens to hold.  See the scan in `main` for which tree each one
    # excludes.
    ap.add_argument("--train-n", type=int, default=1000,
                    help="training draw size of the fixed fleet")
    ap.add_argument("--train-seed", type=int, default=0,
                    help="training draw seed of the fixed fleet")
    ap.add_argument("--lora-rank", type=int, default=16,
                    help="LoRA rank of the fixed fleet")
    ap.add_argument("--lora-init-seed", type=int, default=0,
                    help="LoRA initialisation seed of the fixed fleet")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--limit-points", type=int, default=None,
                    help="score only the first N query sets (a timing probe)")
    ap.add_argument("--out", default="qmix_scores.csv")
    ap.add_argument("--geometry-out", default="qmix_geometry.csv",
                    help="MDS coordinates for the pooled/overlay figure")
    ap.add_argument("--no-geometry", action="store_true",
                    help="scores only; skip the MDS fits and alignments")
    # Seeds are deliberately *discovered* rather than assumed.  The tree emits
    # ten draw seeds per composition as ten independent jobs, and this sweep is
    # expected to be re-run as more of them land -- so the default is "score
    # whatever has been extracted, and say what was not".  Naming seeds
    # explicitly turns an absent one back into an error.
    ap.add_argument("--seed", action="append", dest="seeds", type=int,
                    help="repeatable; restrict to these draw seeds, and treat "
                         "an unextracted one as fatal. Default: every seed the "
                         "spec defines that is present in the cache.")
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
    if args.seeds:
        unknown = sorted(set(args.seeds) - set(qmix.seeds))
        if unknown:
            raise SystemExit(
                f"seed(s) {unknown} are not in the {args.qmix_spec} spec, which "
                f"defines {list(qmix.seeds)}. A seed outside the spec has no "
                f"config and so no draw to score.")
        draws = [d for d in draws if d["seed"] in set(args.seeds)]
    if args.limit_points:
        draws = draws[:args.limit_points]
    n_seeds = len({d["seed"] for d in draws})
    print(f"{len(draws)} query sets over {len(qmix.points())} compositions "
          f"x {n_seeds} draw seeds"
          f"{'' if args.seeds else f' (of {len(qmix.seeds)} in the spec)'}",
          flush=True)

    spec = SPECS[args.spec]
    wanted = {tuple(m) for m in spec.mixture_pcts()}

    # No draw filter on the scan.  The `behavioral_repr` / `functional_repr`
    # tokens report on ONE draw, and this experiment has 150 -- so a filter here
    # could only be right for one point.
    #
    # The *training* pin, by contrast, is not optional and is four fields wide.
    # One content-addressed cache holds every tree this project has run, and four
    # of them are also yahoo, also this base model and also on the 16-point 25%
    # grid -- they differ only in the fields below:
    #
    #   n_samples, seed  the nsweep's 90 further draws of yahoo, and the
    #                    1004-adapter group-size pool
    #   lora_rank        the rsweep's r1 ... r128
    #   lora_init_seed   the initsweep's i00 ... i09
    #
    # Without all four the scan returns 1700 adapters rather than 16 and the
    # completeness guard below fires on every point.  The mixture filter is a
    # fifth condition and is applied by weight rather than by name.
    index = scan_cache(cache_root, base_model_id=args.base_model,
                       datasets=["yahoo"])
    index = index.filter(n_samples=args.train_n, seed=args.train_seed,
                         lora_rank=args.lora_rank,
                         lora_init_seed=args.lora_init_seed)
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

    # Draws are grouped by composition rather than scored flat, because the
    # pooled matrix and the overlay are properties of a *point* -- all of its
    # seeds at once -- and not of any one draw.
    by_point = {}
    for d in draws:
        by_point.setdefault((d["arm"], d["yahoo_pct"]), []).append(d)

    rows, geo_rows, absent = [], [], []
    t_start = time.time()
    for name in names:
        base_spec = specs[name]
        for (arm, pct), group in by_point.items():
            per_seed_dm = {}
            for d in group:
                spec_d = with_draw(base_spec, d["draw"]) \
                    if name in QUERY_DEPENDENT else dict(base_spec)
                have = (name not in QUERY_DEPENDENT
                        or draw_available(spec_d["taxonomy"], cache_root,
                                          args.base_model, ids, d["draw"]))
                if not have:
                    # An explicitly named seed that has not been extracted is a
                    # typo or a job that failed, and either way the caller asked
                    # for something that is not there.
                    if args.seeds:
                        raise SystemExit(
                            f"seed {d['seed']} of {arm} at yahoo={pct}% has no "
                            f"{spec_d['taxonomy']} extraction for all "
                            f"{len(ids)} models. It was named on the command "
                            f"line, so this is fatal; drop --seed to score "
                            f"whatever is present instead.")
                    absent.append((name, arm, pct, d["seed"]))
                    continue
                t0 = time.time()
                dm = point_matrix(index, ids, cache_root, name, spec_d,
                                  use_cache=not args.no_cache)
                dcor, disp = score_point(dm, ids, vertex_names)
                rows.append({
                    "perspective": name, "taxonomy": spec_d["taxonomy"],
                    "metric": spec_d["metric"], "arm": arm, "yahoo_pct": pct,
                    "seed": d["seed"],
                    "recipe_hash": d["draw"]["recipe_hash"],
                    "n_models": len(ids), "dcor": dcor, "disparity": disp,
                })
                per_seed_dm[d["seed"]] = dm
                print(f"  {name:<20} {arm:<9} yahoo={pct:<4} "
                      f"s={d['seed']:<3} dcor={dcor:.4f} disp={disp:.4f} "
                      f"{time.time() - t0:6.1f}s", flush=True)

            if not per_seed_dm or args.no_geometry:
                continue

            # The pooled read: distances averaged over the seeds present, then
            # scored and embedded once.  Marked `seed = POOLED` rather than left
            # out, so a driver can draw the summary and the spread from one file
            # -- but it is NOT a seed, and anything computing a median or a band
            # over seeds has to exclude it.
            pooled, aligned, pooled_dm = point_geometry(per_seed_dm, ids)
            dcor, disp = score_point(pooled_dm, ids, vertex_names)
            ref = next(iter(per_seed_dm.values()))
            rows.append({
                "perspective": name, "taxonomy": ref.taxonomy,
                "metric": ref.metric, "arm": arm, "yahoo_pct": pct,
                "seed": POOLED, "recipe_hash": group[0]["draw"]["recipe_hash"],
                "n_models": len(ids), "dcor": dcor, "disparity": disp,
            })
            print(f"  {name:<20} {arm:<9} yahoo={pct:<4} pooled "
                  f"dcor={dcor:.4f} disp={disp:.4f}  "
                  f"({len(per_seed_dm)} seed(s))", flush=True)

            spec_g = {"taxonomy": ref.taxonomy, "metric": ref.metric}
            for kind, geo in pooled.items():
                geo_rows += geometry_rows(geo, name, spec_g, arm, pct,
                                          kind, POOLED)
            for seed, geo in aligned.items():
                geo_rows += geometry_rows(geo, name, spec_g, arm, pct,
                                          "seed_aligned", seed)

    out = Path(args.outdir) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(out, FIELDS, rows)
    print(f"\nwrote {len(rows)} rows to {out} in "
          f"{time.time() - t_start:.0f}s", flush=True)
    if geo_rows:
        gout = Path(args.outdir) / args.geometry_out
        write_csv(gout, GEOMETRY_FIELDS, geo_rows)
        print(f"wrote {len(geo_rows)} coordinate rows to {gout}", flush=True)

    if absent:
        seeds = sorted({a[3] for a in absent})
        print(f"\nskipped {len(absent)} (perspective, point, seed) "
              f"combinations with no extraction; seeds affected: {seeds}.",
              flush=True)
        print("  These are query sets the tree emitted but the cluster has not "
              "run. Re-running this sweep once they land picks them up with no "
              "change here.", flush=True)


if __name__ == "__main__":
    main()
