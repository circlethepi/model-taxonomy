#!/usr/bin/env python
"""Compute the **label null** for every cell ``figures/figure2_v2`` plots.

A **label null** keeps both real geometries and permutes which model is which,
destroying only the model-to-model correspondence.  It is the counterpart to the
**structure null** of :mod:`src.analysis.baselines`, which instead replaces the
taxonomy with a structureless configuration and keeps the truth.  The figure
artifact that displays it is the **permutation band**; ``docs/terminology.md``
records that the figures' "permutation null" and the note's "label null" are one
concept.  The two statistics are :func:`src.analysis.configurations.protest` and
:func:`src.analysis.matrices.dcor_test`.

**This script edits nothing upstream, by construction.**  The two sweeps that
produce figure 2's bottom row write CSVs whose schemas — and whose readers'
validators — have no p-value column, and putting one there would mean touching
``_SCORE_FIELDS``, ``select_score`` and ``score_field`` and re-scoring twelve
suites.  Instead this imports the sweep drivers *as libraries* and re-derives
the objects they throw away.  Both are clean modules (``if __name__ ==
"__main__"`` guards, argparse inside ``main()``, no import-time work), and
``sweep_nsweep`` already imports ``sweep_group_size`` this way.

``score_group`` and ``score_slice`` return bare floats, so they cannot be reused
directly — but everything *above* them can, and the four lines that build the
objects are mirrored here against the same ``MDS_SEED`` and the same
``n_components``.  That is deliberate and testable: the observed statistic this
script computes must equal the sweep's ``disparity_B`` / ``disparity_requested``
for the same cell, or it is measuring something the figure does not plot.

**Two outputs, for two different readers.**

* Every null is stored in ``07A_permutation_tests``
  (:class:`src.cache.permutation_cache.PermutationCache`), keyed by the sorted
  model set and the test parameters.  That is the durable copy, it makes the
  script resumable — a cell already present is skipped — and it is where a
  later experiment finds these numbers.
* A small digest goes to ``results/figure2_v2/permutation_null.json``, holding
  only the pooled band quantiles the figure reads, so a driver opens one file
  rather than thousands of cache entries.  ``results`` is gitignored, so this is
  a generated artifact like ``results/baselines/constants.json``.

**``--dcor`` is off by default.**  ``dcor_test`` copies an n x n matrix per
permutation draw, so at n=500 it costs tens of seconds per cell — and figure 2
plots Procrustes disparity on every panel and dCor on none, so the output would
be written and never read.  The path is built and tested so that a later dCor
figure turns it on rather than building it.

Usage::

    python scripts/make_permutation_null.py --probe            # measure, then stop
    python scripts/make_permutation_null.py --paths bars,nsweep
    python scripts/make_permutation_null.py --replicates 10
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis.baselines import BAND_Q                          # noqa: E402
from src.analysis.bridge import fit_geometry                       # noqa: E402
from src.analysis.configurations import protest                    # noqa: E402
from src.analysis.ground_truth import (simplex_distance_matrix,    # noqa: E402
                                       simplex_geometry)
from src.analysis.matrices import dcor_test                        # noqa: E402
from src.cache import PermutationCache                             # noqa: E402
from src.plots import simplex_suite as suite                       # noqa: E402
from src.plots import simplex_runs as runs                         # noqa: E402

FIGURES_ROOT = REPO / "figures"
COLLECTION_DIR = FIGURES_ROOT / "simplex_collection_size"
NSWEEP_DIR = FIGURES_ROOT / "simplex3_nsweep_olmo2_nsweep"

#: Where the digest lands.  Beside ``results/baselines/constants.json`` and
#: generated the same way: gitignored, rebuilt by naming this script.
OUT = REPO / "results/figure2_v2/permutation_null.json"

#: The base model whose corpora panel 2 compares.  Panel 1 compares the four
#: base models on yahoo; together that is six distinct runs, not twelve.
PANEL2_BASE_MODEL = "allenai/OLMo-2-0425-1B-Instruct"

#: The training draw, LoRA rank and LoRA init seed every simplex3 suite
#: predating the sweeps was built at.  The per-suite drivers do not declare
#: these — they did not have to when they were written — but the nsweep has
#: since put 90 further draws of yahoo under this base model, the rank sweep
#: trained the same mixtures at eight ranks and the init sweep trained them from
#: ten draws of ``A``, so a scan filtered only by corpus and mixture now returns
#: far more models than the suite has and ``n_expected`` trips.
#: ``figures/figure2`` pins all three for the one run it builds; this pins them
#: for all six, and a driver that declares its own still wins.
DEFAULT_TRAIN_DRAW = (1000, 0)
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_INIT_SEED = 0

#: The suite's embedder hashes **as imported**, captured before anything runs.
#:
#: ``run_suite`` declares ``global EMBEDDER, DATASET_EMBEDDER`` and assigns them
#: only when the caller passes a value, so a corpus that has its own — dolly and
#: oasst1 do — leaves them set for every run afterwards. This script builds six
#: runs in one process, and dolly sorts first, so a yahoo run following it looked
#: up yahoo adapters under dolly's embedder and failed with "no behavioral
#: representation". Passing these explicitly whenever a driver declares none
#: makes each run independent of the ones before it.
SUITE_EMBEDDER = suite.EMBEDDER
SUITE_DATASET_EMBEDDER = suite.DATASET_EMBEDDER

#: figure 2 plots the **requested** truth on panel 4 (``disparity_requested``),
#: so that is the truth the nsweep null is built against.  ``realized`` exists
#: and differs — largest-remainder allocation makes a requested 25/75 realize as
#: 3/7 at N=10 — but nothing on this figure reads it.
NSWEEP_TRUTH = "requested"


# ── the two statistics ────────────────────────────────────────────────────────

def protest_params(n_components: int, n_permutations: int, seed: int) -> dict:
    """Everything about a PROTEST run that changes its numbers.

    Including ``n_components``: the same two matrices compared through a 2-D and
    a 3-D embedding are different measurements, and a key that omitted it would
    let them collide.
    """
    return {"n_permutations": n_permutations, "random_state": seed,
            "n_components": n_components, "scaling": True, "reflection": True}


def dcor_params(n_permutations: int, seed: int) -> dict:
    """Everything about a dCor permutation run that changes its numbers.

    No ``n_components``: ``dcor_test`` works on the distance matrices directly,
    with no embedding step to distort.
    """
    return {"n_permutations": n_permutations, "random_state": seed,
            "bias_corrected": True}


def run_cell(cache, *, members, source_handle, truth_kind, dm, tdm, tgeo,
             n_permutations, seed, with_dcor, label=None, provenance=None):
    """Both tests on one cell, reading through the cache.  Returns the configs.

    A cell already stored is **not** recomputed — which is what makes a job
    resumable after a timeout, and what makes a second figure asking for the
    same cells free.
    """
    out = {}
    truth_dim = int(np.asarray(tgeo.coordinates).shape[1])
    wanted = [("protest", protest_params(truth_dim, n_permutations, seed))]
    if with_dcor:
        wanted.append(("dcor", dcor_params(n_permutations, seed)))

    for test, params in wanted:
        gk = cache.group_key(members=members, source_handle=source_handle,
                             truth_kind=truth_kind)
        tk = cache.test_key(members=members, source_handle=source_handle,
                            truth_kind=truth_kind, test=test, params=params)
        hit = cache.load_result(test, gk, tk)
        if hit is not None:
            out[test] = hit
            continue

        if test == "protest":
            # Same MDS seed and the same n_components the sweeps use, so the
            # observed disparity reproduces the CSV rather than merely
            # resembling it.  _pad_geometry is a no-op when the fit already has
            # the truth's width, but stating it means a future change to either
            # dimension fails loudly instead of silently zero-padding.
            geo = fit_geometry(dm, method="mds", n_components=truth_dim,
                               random_state=suite.MDS_SEED)
            res = protest(tgeo, suite._pad_geometry(geo, truth_dim),
                          n_permutations=n_permutations, random_state=seed)
            statistic, exact = res.disparity, False
        else:
            res = dcor_test(dm, tdm, n_permutations=n_permutations,
                            random_state=seed)
            statistic, exact = res.statistic, res.exact

        cache.save_result(
            test=test, members=members, source_handle=source_handle,
            truth_kind=truth_kind, params=params, statistic=statistic,
            p_value=res.p_value, null=res.null, n_models=res.n_models,
            n_permutations=res.n_permutations, exact=exact, label=label,
            provenance=provenance or {})
        out[test] = cache.load_result(test, gk, tk)
    return out


# ── accumulating the digest ───────────────────────────────────────────────────

class Digest:
    """Pooled null draws per figure key, reduced to band quantiles at the end.

    Pooling rather than averaging the per-cell bands: quantiles of a mixture are
    not the mixture of the quantiles, and the band is meant to describe what the
    whole group of cells could have scored.
    """

    def __init__(self) -> None:
        self._draws: dict[str, list[np.ndarray]] = defaultdict(list)
        self._meta: dict[str, dict] = {}

    def add(self, key: str, config: dict, **meta) -> None:
        self._draws[key].append(np.asarray(config["null"], dtype=np.float64))
        rec = self._meta.setdefault(key, {"n_cells": 0, "p_values": [],
                                          "statistics": []})
        rec["n_cells"] += 1
        rec["p_values"].append(float(config["p_value"]))
        rec["statistics"].append(float(config["statistic"]))
        rec.update(meta)

    def entries(self) -> dict:
        out = {}
        for key, chunks in self._draws.items():
            pooled = np.concatenate(chunks)
            lo, hi = (float(np.percentile(pooled, q)) for q in BAND_Q)
            rec = dict(self._meta[key])
            rec.update({
                "lo": lo, "mid": float(np.median(pooled)), "hi": hi,
                "n_draws": int(pooled.size),
                "p_min": min(rec["p_values"]), "p_max": max(rec["p_values"]),
                "statistic_median": float(np.median(rec["statistics"])),
            })
            # The per-cell lists are the reason to keep the cache; the digest is
            # meant to stay small enough to read.
            rec.pop("p_values"), rec.pop("statistics")
            out[key] = rec
        return out


# ── path: the bar panels ──────────────────────────────────────────────────────

def _scratch_dir(_made=[]) -> str:
    """A throwaway directory for anything ``run_suite`` insists on writing."""
    import tempfile

    if not _made:
        _made.append(tempfile.mkdtemp(prefix="permnull-"))
    return _made[0]


def load_driver(figure_dir: str):
    """A figure driver's module constants, without running it.

    The per-run parameters — draw hashes, embedder hashes, ``N_EXPECTED``,
    ``SELECT`` — exist only as module constants in twelve separate
    ``make_figures.py`` files; ``simplex_runs.RUNS`` carries just
    ``(base_model, corpus, figure_dir)``.  Loading by path under a unique module
    name is the read-only way to get at them; importing them all as
    ``make_figures`` would collide.
    """
    path = FIGURES_ROOT / figure_dir / "make_figures.py"
    spec = importlib.util.spec_from_file_location(f"_drv_{figure_dir}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bars_cells(run, cache_root, no_cache, levels=None):
    """``({level: DistanceMatrix}, ids)`` for one run, from its own constants.

    *levels* restricts which of the driver's levels are asked for, so one level
    whose representations are missing from the cache does not cost the run its
    other three — see :func:`do_bars`.
    """
    drv = load_driver(run.figure_dir)
    select = {lv: [(surr, metric)] for lv, pairs in drv.SELECT.items()
              for surr, metric in pairs}
    if levels is not None:
        select = {k: v for k, v in select.items() if k in levels}
    kwargs = dict(
        # **Never the suite's own directory.** ``crosslevel_only=True`` with
        # ``closer=False`` means nothing should be written, but "should" is not
        # a guarantee, and this script's whole claim is that it leaves the
        # twelve suites untouched. Pointing outdir at a scratch directory makes
        # that structural rather than a property of flags held right.
        base_model=drv.BASE_MODEL, draw=drv.DRAW, outdir=_scratch_dir(),
        cache_root=cache_root, levels=list(select), no_cache=no_cache,
        surrogates=False, select=select, crosslevel_only=True, closer=False,
        datasets=getattr(drv, "DATASETS", [run.corpus]),
        source="scripts/make_permutation_null.py",
    )
    for name, flag in (("skip_sweep", True), ("skip_detail", True)):
        kwargs[name] = flag
    # The training-draw and rank filters apply to **yahoo only**. That is the
    # one pool the nsweep and the rank sweep added to, so it is the only one
    # where corpus + mixture is ambiguous; dolly and oasst1 have exactly their
    # 35 adapters and nothing else. Applying the defaults everywhere is not
    # merely redundant there, it is wrong — oasst1 was trained at n500, so
    # train_draw=(1000, 0) selects none of it.
    disambiguate = run.corpus == "yahoo"
    for name, attr, default in (
            ("n_expected", "N_EXPECTED", None),
            ("embedder", "EMBEDDER", SUITE_EMBEDDER),
            ("dataset_embedder", "DATASET_EMBEDDER", SUITE_DATASET_EMBEDDER),
            ("mixtures", "MIXTURES", None),
            ("train_draw", "TRAIN_DRAW",
             DEFAULT_TRAIN_DRAW if disambiguate else None),
            ("lora_rank", "LORA_RANK",
             DEFAULT_LORA_RANK if disambiguate else None),
            ("lora_init_seed", "LORA_INIT_SEED",
             DEFAULT_LORA_INIT_SEED if disambiguate else None)):
        value = getattr(drv, attr, default)
        if value is not None:
            kwargs[name] = value

    accepted = set(inspect.signature(suite.run_suite).parameters)
    rank = kwargs.pop("lora_rank", None) if "lora_rank" not in accepted else None
    for unsupported in sorted(set(kwargs) - accepted):
        print(f"    note: run_suite here takes no {unsupported!r}; dropping it",
              flush=True)
        kwargs.pop(unsupported)

    with _rank_filter(rank):
        per_level, ids = suite.run_suite(**kwargs)
    return _unwrap_cells(per_level), ids


@contextmanager
def _rank_filter(lora_rank):
    """Apply ``lora_rank`` to ``run_suite``'s scan on a branch that lacks it.

    ``run_suite`` gained a ``lora_rank`` parameter with the LoRA-rank sweep,
    which is newer than this branch; ``CacheIndex.filter(lora_rank=...)`` and
    ``CacheEntry.lora_rank`` are both already here, so only the plumbing is
    missing.  Without the filter a scan matched on corpus, draw and mixture
    finds the same 16 mixtures at eight ranks — 128 models — and ``n_expected``
    trips, which is the guard working correctly and the bar panels not getting
    built.

    Wrapping the scan is preferred over adding the parameter to
    ``simplex_suite`` here: that module is shared, the addition already exists
    upstream, and two independent versions of one parameter is how the two
    branches would quietly stop agreeing.  This applies exactly what upstream
    applies — ``idx.filter(lora_rank=...)``, in the same position, before the
    mixture filter — and becomes dead code the moment ``run_suite`` accepts the
    argument, because then it is never called with a rank.
    """
    if lora_rank is None:
        yield
        return
    original = suite.scan_cache

    def _scanned(*a, **kw):
        return original(*a, **kw).filter(lora_rank=lora_rank)

    suite.scan_cache = _scanned
    try:
        yield
    finally:
        suite.scan_cache = original


def _unwrap_cells(per_level):
    """``{level: DistanceMatrix}`` for the levels that built exactly one cell.

    A level whose grid came back empty, or with more than one buildable cell,
    is dropped rather than guessed at: a null computed from the wrong
    perspective would look entirely reasonable and mean something else.
    """
    cells = {}
    for level, grid in per_level.items():
        got = [c for c in grid.values()
               if c is not None and not isinstance(c, str)]
        if len(got) == 1:
            cells[level] = got[0]
    return cells


def _driver_levels(run) -> list[str]:
    """The levels this run's driver selects, in its own order."""
    return list(load_driver(run.figure_dir).SELECT)


def panels_of(run) -> list[str]:
    """Which bar panels of figure 2 this run appears in — it may be both.

    Panel 1 is every base model on yahoo; panel 2 is every corpus on OLMo-2-1B.
    ``simplex3_olmo2`` is the cell where they cross, and it is drawn on both, so
    its null belongs to both pools.
    """
    out = []
    if run.corpus == "yahoo":
        out.append("models")
    if run.base_model == PANEL2_BASE_MODEL:
        out.append("corpora")
    return out


def do_bars(cache, digest, args, cache_root):
    """Panels 1-2: six distinct runs, four levels each."""
    wanted = {r.figure_dir: r for r in runs.runs_for_corpus("yahoo")}
    wanted.update({r.figure_dir: r for r in runs.runs_for_model(PANEL2_BASE_MODEL)})
    if args.runs:
        unknown = set(args.runs) - set(wanted)
        if unknown:
            raise SystemExit(f"unknown run dir(s) {sorted(unknown)}; "
                             f"figure 2 draws {sorted(wanted)}")
        wanted = {k: v for k, v in wanted.items() if k in args.runs}
    for run in sorted(wanted.values(), key=lambda r: r.figure_dir):
        t0 = time.time()
        # Two of the canonical surrogates name "the last layer", a position in a
        # stack rather than a number, so the architecture must be bound first.
        suite.apply_architecture(suite.architecture(run.base_model))
        try:
            cells, ids = bars_cells(run, cache_root, args.no_cache)
        except Exception as exc:                       # noqa: BLE001
            # One level can fail on its own — several base models have no
            # stored behavioral embeddings for some adapters, and that raises
            # for the whole call. Retry dropping one level at a time so the run
            # keeps the three that are fine rather than losing all four to one
            # gap. Dropping rather than asking level by level, because
            # ``crosslevel_only`` needs at least two panels and refuses a
            # single-level build outright.
            print(f"  {run.figure_dir:<26} all-levels build failed "
                  f"({type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:80]}); retrying without one",
                  flush=True)
            all_levels = _driver_levels(run)
            cells, ids = {}, None
            for drop in all_levels:
                keep = {lv for lv in all_levels if lv != drop}
                try:
                    cells, ids = bars_cells(run, cache_root, args.no_cache,
                                            levels=keep)
                except Exception:                      # noqa: BLE001
                    continue
                print(f"      built without {drop!r}: "
                      f"{sorted(cells)}", flush=True)
                break
            if not cells or ids is None:
                print(f"  {run.figure_dir:<26} no subset built; skipped",
                      flush=True)
                continue
        tdm, tgeo = suite.truth_dm(ids), suite.truth_geometry(ids)
        for level, dm in sorted(cells.items()):
            got = run_cell(
                cache, members=ids, source_handle=f"bars/{run.figure_dir}/{level}",
                truth_kind="simplex", dm=dm, tdm=tdm, tgeo=tgeo,
                n_permutations=args.n_permutations, seed=args.seed,
                with_dcor=args.dcor, label=f"{run.label} · {level}",
                provenance={"path": "bars", "figure_dir": run.figure_dir,
                            "level": level, "corpus": run.corpus,
                            "base_model": run.base_model})
            # Three keys per cell, from one null.  The per-cell key serves
            # ``--perm-grain bar``; the two panel keys serve the default
            # per-level-group band, and they are accumulated here rather than
            # combined in the driver because quantiles of a mixture are not the
            # mixture of the quantiles — a band pooled from the stored lo/hi
            # would be a different number than a band pooled from the draws.
            digest.add(f"bars|{run.figure_dir}|{level}", got["protest"],
                       n_models=len(ids), corpus=run.corpus,
                       base_model=run.base_model, level=level)
            for panel in panels_of(run):
                digest.add(f"barlevel|{panel}|{level}", got["protest"],
                           n_models=len(ids), level=level, panel=panel)
        print(f"  {run.figure_dir:<26} {len(cells)} level(s), {len(ids):>3} "
              f"models  {time.time() - t0:6.1f}s", flush=True)


# ── path: the nsweep panel ────────────────────────────────────────────────────

def do_nsweep(cache, digest, args, cache_root):
    """Panel 4: every (perspective, n_samples, seed) slice, all at n=16."""
    sys.path.insert(0, str(NSWEEP_DIR))
    sys.path.insert(0, str(COLLECTION_DIR))
    import sweep_nsweep as sw                                       # noqa: E402
    from sweep_group_size import canonical_perspectives             # noqa: E402
    from src.analysis.discovery import scan_cache, CacheIndex       # noqa: E402
    from src.experiments.data_simplex_spec import SPECS             # noqa: E402
    from src.plots.simplex import raw_mixture_pcts, sort_by_mixture  # noqa: E402

    suite.apply_architecture(suite.architecture(sw.BASE_MODEL))
    spec = SPECS["yahoo_nsweep"]
    wanted = {tuple(m) for m in spec.mixture_pcts()}
    index = scan_cache(cache_root, base_model_id=sw.BASE_MODEL,
                       behavioral_draw=suite.DRAW, functional_draw=suite.DRAW,
                       datasets=["yahoo"])
    index = CacheIndex([e for e in index.entries
                        if raw_mixture_pcts(e.model_id) in wanted],
                       index.cache_root)
    slices = index.slices(by=("n_samples", "seed"))
    bought = set(spec.extract_sizes or spec.train_sizes)
    slices = {k: v for k, v in slices.items() if k[0] in bought}

    specs = canonical_perspectives()
    names = args.perspectives or list(specs)
    keys = sorted(slices)
    if args.limit_slices:
        keys = keys[:args.limit_slices]

    for name in names:
        base_spec = specs[name]
        t0 = time.time()
        for (n_samples, seed_) in keys:
            sub = slices[(n_samples, seed_)]
            ids = sort_by_mixture(list(sub.model_ids))
            vertex_names = [f"g{j + 1}"
                            for j in range(suite.truth_weights(ids).shape[1])]
            s = dict(base_spec)
            # The dataset level's surrogate IS the training draw, so its
            # selector moves with the slice; every other level reads the fixed
            # query draw and is slice-independent.
            if "dataset_selector" in s:
                s["dataset_selector"] = {**s["dataset_selector"],
                                         "n_samples": n_samples, "seed": seed_}
            try:
                dm = sw.slice_matrix(sub, ids, cache_root, name, s,
                                     use_cache=not args.no_cache)
            except Exception as exc:                   # noqa: BLE001
                print(f"    {name} n={n_samples} s={seed_} skipped — "
                      f"{type(exc).__name__}: {exc}", flush=True)
                continue
            W = suite.truth_weights(ids)
            tdm = simplex_distance_matrix(W, list(ids), vertex_names)
            tgeo = simplex_geometry(W, list(ids), vertex_names)
            got = run_cell(
                cache, members=ids,
                source_handle=f"nsweep/{name}/{n_samples}/{seed_}",
                truth_kind=NSWEEP_TRUTH, dm=dm, tdm=tdm, tgeo=tgeo,
                n_permutations=args.n_permutations, seed=args.seed,
                with_dcor=args.dcor, label=f"nsweep {name} n={n_samples}",
                provenance={"path": "nsweep", "perspective": name,
                            "n_samples": n_samples, "seed": seed_,
                            "taxonomy": s["taxonomy"]})
            digest.add(f"nsweep|{name}", got["protest"], n_models=len(ids),
                       perspective=name, taxonomy=s["taxonomy"])
            # Panel 4 draws four level curves against one band, so the pooled
            # key is the one it reads; the per-perspective key is kept for a
            # figure that wants to separate them.
            digest.add("nsweep|ALL", got["protest"], n_models=len(ids))
        print(f"  {name:<20} {len(keys)} slice(s)  "
              f"{time.time() - t0:6.1f}s", flush=True)


# ── path: the collection-size panel ───────────────────────────────────────────

def do_collection(cache, digest, args, cache_root, *, probe=False):
    """Panel 3: (perspective, n, replicate) groups drawn off the pool matrix."""
    sys.path.insert(0, str(COLLECTION_DIR))
    import sweep_group_size as sg                                   # noqa: E402
    from src.analysis.discovery import scan_cache                   # noqa: E402
    from src.plots.simplex import mixture_weights, sort_by_mixture   # noqa: E402

    # The sweep takes its query draw from ``suite.DRAW``'s own defaults (its
    # ``--draw-*`` flags default to them), so leaving it untouched here is what
    # reproduces the groups it scored, not an omission.
    suite.apply_architecture(suite.architecture(sg.BASE_MODEL))
    index = scan_cache(cache_root, base_model_id=sg.BASE_MODEL,
                       behavioral_draw=suite.DRAW, functional_draw=suite.DRAW,
                       datasets=["yahoo"])
    index = sg.pin_pool(index, train_n=1000, train_seed=0,
                        exclude=sg.SIMPLEX3_EXTRA)
    ids = sort_by_mixture([e.model_id for e in index.entries])
    weights = np.vstack([mixture_weights(m) for m in ids])
    vertex_names = [f"g{j + 1}" for j in range(weights.shape[1])]
    weights_of = {m: w for m, w in zip(ids, weights)}
    refs = sg.reference_ids(ids, weights)
    pool = [m for m in ids if m not in set(refs)]

    specs = sg.canonical_perspectives()
    names = args.perspectives or list(specs)
    if probe:
        names = names[:1]
    n_grid = [int(x) for x in args.n_grid.split(",") if x.strip()]
    replicates = 3 if probe else args.replicates

    print(f"pool: {len(pool)} sampled + {len(refs)} reference "
          f"= {len(ids)} models; {len(names)} perspective(s)", flush=True)
    matrices = sg.pool_matrices(index, ids, cache_root, names,
                                use_cache=not args.no_cache)

    timings = []
    for n in n_grid:
        if n > len(pool):
            print(f"  n={n} skipped: larger than the {len(pool)}-model pool")
            continue
        drawn = sg.groups_for(pool, n, replicates, args.group_seed)
        for pname, dm in matrices.items():
            t0 = time.time()
            seen = set()
            for group, shuffle, _n_disjoint in drawn:
                key = frozenset(group)
                if key in seen:       # the same models twice is the same null
                    continue
                seen.add(key)
                W = np.vstack([weights_of[m] for m in group])
                tdm = simplex_distance_matrix(W, list(group), vertex_names)
                tgeo = simplex_geometry(W, list(group), vertex_names)
                try:
                    got = run_cell(
                        cache, members=list(group),
                        source_handle=f"collection/{pname}",
                        truth_kind="simplex", dm=dm.reindex(list(group)),
                        tdm=tdm, tgeo=tgeo,
                        n_permutations=args.n_permutations, seed=args.seed,
                        with_dcor=args.dcor,
                        label=f"collection {pname} n={n}",
                        provenance={"path": "collection", "perspective": pname,
                                    "n": n, "shuffle_id": shuffle,
                                    "taxonomy": specs[pname]["taxonomy"]})
                except Exception as exc:               # noqa: BLE001
                    print(f"    {pname} n={n} skipped — "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    continue
                digest.add(f"collection|{pname}|{n}", got["protest"],
                           n_models=n, perspective=pname, n_requested=n,
                           taxonomy=specs[pname]["taxonomy"])
                # Panel 3's band is a curve in n drawn behind four level
                # curves, so it pools the perspectives at each n.
                digest.add(f"collection|ALL|{n}", got["protest"],
                           n_models=n, n_requested=n)
            dt = time.time() - t0
            timings.append((pname, n, len(seen), dt))
            print(f"  {pname:<20} n={n:<5} {len(seen):>3} distinct group(s)  "
                  f"{dt:7.1f}s", flush=True)
    return timings


def report_probe(timings, args) -> None:
    """What the full run would cost, from what the probe actually measured."""
    print("\n── probe ────────────────────────────────────────────────────")
    print(f"{'perspective':<22}{'n':>6}{'groups':>8}{'s/group':>10}{'total s':>10}")
    per_group = {}
    for pname, n, k, dt in timings:
        rate = dt / max(k, 1)
        per_group[n] = max(per_group.get(n, 0.0), rate)
        print(f"{pname:<22}{n:>6}{k:>8}{rate:>10.2f}{dt:>10.1f}")
    n_persp = 8
    print("\nextrapolated full run, per replicate count "
          f"({n_persp} perspectives, --dcor {'on' if args.dcor else 'off'}):")
    for reps in (10, 25, 50, 100):
        total = sum(rate * min(reps, 1e9) * n_persp
                    for rate in per_group.values())
        print(f"  --replicates {reps:>4}   ≈ {total / 60:8.1f} min "
              f"({total / 3600:5.2f} h) of permutation work")
    print("\nThis excludes the pool load, which is measured above as part of "
          "pool_matrices and is the same whatever the replicate count.\n"
          "Choose --replicates, then rerun without --probe.")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paths", default="bars,nsweep,collection",
                    help="comma-separated subset of bars,nsweep,collection")
    ap.add_argument("--probe", action="store_true",
                    help="measure one perspective x 3 replicates on the "
                         "collection path, report the extrapolated cost, and "
                         "stop without writing the digest")
    ap.add_argument("--dcor", action="store_true",
                    help="also run dcor_test (default off: it is O(n^2) per "
                         "permutation draw and figure 2 has no dCor panel)")
    ap.add_argument("--replicates", type=int, default=10,
                    help="collection-path replicates per (perspective, n)")
    ap.add_argument("--n-grid", default="5,10,20,50,100,200,500")
    ap.add_argument("--n-permutations", type=int, default=9999)
    ap.add_argument("--seed", type=int, default=0,
                    help="random_state for both tests; part of the cache key")
    ap.add_argument("--group-seed", type=int, default=0,
                    help="the sweep's own shuffle seed, so the groups drawn "
                         "here are the groups it scored")
    ap.add_argument("--perspective", action="append", dest="perspectives")
    ap.add_argument("--run", action="append", dest="runs",
                    help="repeatable figure directory, e.g. simplex3_olmo2; "
                         "default is all six runs figure 2 draws")
    ap.add_argument("--limit-slices", type=int, default=0)
    ap.add_argument("--cache-root", default=None)
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore the shared cache when building matrices")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    cache_root = Path(args.cache_root).expanduser().resolve() \
        if args.cache_root else suite.CACHE_ROOT
    if not Path(cache_root).exists():
        raise SystemExit(
            f"no cache at {cache_root}. Pass --cache-root; from a git worktree "
            "the default is derived from the module's location and may not "
            "resolve to the checkout that holds the cache.")
    cache = PermutationCache(cache_root)
    digest = Digest()
    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    t_start = time.time()

    print(f"cache: {cache_root}")
    print(f"tests: protest{' + dcor' if args.dcor else ''}  "
          f"({args.n_permutations} permutations, random_state={args.seed})\n")

    if args.probe:
        timings = do_collection(cache, digest, args, cache_root, probe=True)
        report_probe(timings, args)
        return 0

    if "bars" in paths:
        print("── bars (panels 1-2) ──")
        do_bars(cache, digest, args, cache_root)
    if "nsweep" in paths:
        print("── nsweep (panel 4) ──")
        do_nsweep(cache, digest, args, cache_root)
    if "collection" in paths:
        print("── collection (panel 3) ──")
        do_collection(cache, digest, args, cache_root)

    entries = digest.entries()
    if not entries:
        raise SystemExit("nothing was scored; the digest would be empty.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/make_permutation_null.py",
        "paths": paths,
        "n_permutations": args.n_permutations,
        "random_state": args.seed,
        "group_seed": args.group_seed,
        "replicates": args.replicates,
        "band_quantiles": list(BAND_Q),
        "with_dcor": args.dcor,
        "cache_root": str(cache_root),
        "entries": entries,
    }
    with out.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")
    print(f"\nwrote {len(entries)} digest entries to {out}  "
          f"(cache hits {cache.hits}, misses {cache.misses}, "
          f"{time.time() - t_start:.0f}s total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
