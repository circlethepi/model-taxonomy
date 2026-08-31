#!/usr/bin/env python
"""Regenerate the whole visualization suite for the simplex3_qwen experiment.

Four taxonomy levels over the 16 Qwen3.5-4B adapters that span a 3-group topic
simplex. Distance matrices in ``copper_r``; MDS embeddings coloured by each
model's own mixture (see :mod:`src.plots.simplex`).

Not every (surrogate, metric) cell exists, and the gaps are structural rather than
accidental:

* **CKA**, **MMD** and **energy** all need more than one row, so none of them can
  run on a ``model mean`` representation, and
  :func:`src.notebook.structure.cka_distance_matrix` takes a single
  (layer, projection), so CKA cannot span a layer grouping.
* **Bures-Wasserstein** stacks per-block factors before its SVD, so every block
  must share an input dim; a selection mixing 2560-input and 4096-input
  projections has no BW value. It is also rank-1, and so carries nothing cosine
  does not, on a single-row representation.

Absent cells are drawn with the reason in place rather than left blank or
silently dropped, so the constraint stays visible in the figure.

Fleet transforms
----------------
Several rows apply a *fleet-level* transform from :mod:`src.analysis.surrogates`
before distancing — centering on the fleet mean, or whitening against the fleet
covariance. These are not alternative metrics; they change what is being
compared. Every level here carries a large component shared by all 16 models
(the same 100 questions, the same Yahoo answer register, the same base model),
and it is identical by construction, so it can only dilute a similarity. The
centered surrogates measure what is left.

Usage
-----
    python scripts/make_simplex3_figures.py                  # everything
    python scripts/make_simplex3_figures.py --level functional
    python scripts/make_simplex3_figures.py --skip-sweep     # omit the slow one
    python scripts/make_simplex3_figures.py --skip-surrogate # raw surrogates only
    python scripts/make_simplex3_figures.py --cache-root PATH  # explicit cache
    python scripts/make_simplex3_figures.py --no-cache          # force a cold run

Reuse
-----
Distance matrices and MDS embeddings are read back from ``06_collections`` in
the shared cache when a previous run has already computed them, and written
there when they have not. ``--no-cache`` ignores what is stored and recomputes
everything, but **still writes the results back** — that is how a warm run is
checked against a cold one: run it once with ``--no-cache`` and once without
against the same ``--cache-root``, and the ``matrix_sha256`` column of
``crosslevel_scores.csv`` must agree between the two. A flag that also skipped
the writes would leave the second run nothing to read, so both runs would
compute cold and agree trivially. See ``docs/notes/caching_collections.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path
from typing import NamedTuple

# ── BLAS threads, pinned before numpy is imported ────────────────────────────
#   This is a large, measured effect, not a precaution. The Bures-Wasserstein
#   precompute takes a thin SVD of a (n_blocks*r, d) matrix — (384, 2560) for a
#   24-block selection. Left to the default thread count on this 14-core host
#   that SVD measured **79 s** idle and 118 s under load; pinned to one thread it
#   measures **5.3 s**, with results identical to floating-point tolerance. The
#   matrices are small enough that threading buys nothing and the synchronisation
#   dominates. Unpinned, the behavioral level did not finish inside 50 minutes.
#
#   Must be set before numpy loads its BLAS, hence the placement above the import
#   rather than in main(). Override with MODEL_TAXONOMY_THREADS if a host differs.
_THREADS = os.environ.get("MODEL_TAXONOMY_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _THREADS)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.analysis import scan_cache  # noqa: E402
from src.analysis.bridge import as_distance_matrix, fit_geometry  # noqa: E402
from src.analysis.comparison import (  # noqa: E402
    _distances, resolve_ordered,
)
from src.analysis.ground_truth import (  # noqa: E402
    dcor_vs_truth, disparity_vs_truth, simplex_distance_matrix, simplex_geometry,
)
from src.analysis.quality import kruskal_stress  # noqa: E402
from src.analysis.surrogates import centered, whitened  # noqa: E402
from src.plots.config import set_style  # noqa: E402
from src.plots.figures import save_figure  # noqa: E402
from src.plots.simplex import (  # noqa: E402
    crosslevel_mds, dm_grid, mds_grid, mixture_label, mixture_weights,
    sort_by_mixture, ternary_legend,
)

# ── Experiment coordinates ────────────────────────────────────────────────────

def _default_cache_root() -> Path:
    """Where the shared cache is, when ``--cache-root`` does not say.

    Every candidate is derived from ``Path(__file__)``, which is why the third
    one is needed: from a git worktree under ``.claude/worktrees/<name>`` the
    first two both point inside the worktree, where no cache has ever been
    written, and the suite silently found nothing to read. The worktree is where
    any work on the caching itself gets done, so that case is resolved here
    rather than papered over with a symlink.
    """
    parts = REPO_ROOT.parts
    candidates = []
    if len(parts) >= 3 and parts[-2] == "worktrees" and parts[-3] == ".claude":
        candidates.append(REPO_ROOT.parents[2] / "results" / "shared_cache")
    candidates += [REPO_ROOT.parent.parent.parent / "results" / "shared_cache",
                   REPO_ROOT / "results" / "shared_cache"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return REPO_ROOT / "results" / "shared_cache"


CACHE_ROOT = _default_cache_root()
ADAPTER_ROOT = CACHE_ROOT / "03_adapters"
#: Which *run* these figures are about.  Run identity belongs on the command
#: line, not in the source: --base-model and the --draw-* flags override both.
#: The defaults are the qwen suite, which is what this script was written for.
BASE_MODEL = "Qwen/Qwen3.5-4B"
BASE_SLUG = BASE_MODEL.replace("/", "--")

#: the question-only query draw both inference stages used
DRAW = {"recipe_hash": "6149cf8055bac2c1", "n_samples": 100, "seed": 1,
        "prompt_format_id": "ea27ccee"}
EMBEDDER = "a3c1f067c13d3cc8"          # nomic-embed-text-v1.5, search_document
DATASET_EMBEDDER = "b37e80a31644dc03"  # authored by the 8B run; recipe-keyed
SAMP_GREEDY, SAMP_SAMPLED = "6f000f01", "58d3f985"
MAX_NEW_TOKENS = 128

#: Architecture.  Read from the checkpoint's own config by
#: :func:`architecture`, not spelled here: a hybrid model's layout is something
#: the config declares (`full_attention_interval`), and hard-coding arithmetic on
#: a literal 4 makes the whole structural section mean the wrong thing for any
#: other model.  These module-level names are the defaults for Qwen3.5-4B and are
#: replaced in ``main`` once the base model is known.
N_LAYERS = 32
FULL_ATTN_LAYERS = [i for i in range(N_LAYERS) if i % 4 == 3]
LINEAR_ATTN_LAYERS = [i for i in range(N_LAYERS) if i % 4 != 3]
ATTN_NUM_HEADS = 16
#: q_proj carries a fused output gate on Qwen3.5, so half its rows are not
#: queries.  Read from the config rather than assumed -- a model without it has
#: no query/gate split to make.
ATTN_OUTPUT_GATE = True
#: hidden state h_{L+1} is the output of transformer layer L; h0 is the embedding
FULL_ATTN_STATES = [L + 1 for L in FULL_ATTN_LAYERS]
LINEAR_ATTN_STATES = [L + 1 for L in LINEAR_ATTN_LAYERS]
N_STATES = N_LAYERS + 1


def architecture(base_model: str) -> dict:
    """Layer count, head count and attention layout, from the checkpoint.

    ``full_attention_interval`` is what makes a Qwen3.5 hybrid: every
    ``interval``-th layer is softmax attention and the rest are gated-delta-rule
    linear attention.  A model that does not declare it has no linear-attention
    layers at all, so ``LINEAR_ATTN_LAYERS`` comes back empty and every
    ``linear-attn`` spec downstream simply does not exist -- the figure set gets
    *smaller*, which is the correct degradation rather than a special case.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    # Some checkpoints ship a composite config (Qwen3.5 is a
    # ForConditionalGeneration whose decoder sits under `text_config`), and the
    # decoder is the part every figure here is about.  get_text_config() is the
    # transformers-supported way to ask for it and returns the config itself when
    # there is no nesting.
    cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    return layout(
        n_layers=int(cfg.num_hidden_layers),
        n_heads=int(cfg.num_attention_heads),
        full_attention_interval=getattr(cfg, "full_attention_interval", None),
        attn_output_gate=bool(getattr(cfg, "attn_output_gate", False)),
    )


def layout(n_layers, n_heads, full_attention_interval=None, attn_output_gate=False):
    """The attention layout, split out so it can be tested without a checkpoint.

    Everything that can regress lives here; :func:`architecture` is only the part
    that reads the four values off a config.
    """
    if full_attention_interval:
        iv = int(full_attention_interval)
        full = [i for i in range(n_layers) if i % iv == iv - 1]
    else:
        full = list(range(n_layers))
    linear = [i for i in range(n_layers) if i not in set(full)]
    return {"n_layers": n_layers, "n_heads": n_heads, "full": full,
            "linear": linear, "output_gate": attn_output_gate}


def apply_architecture(arch: dict) -> None:
    """Install *arch* into the module-level names the spec builders read."""
    global N_LAYERS, FULL_ATTN_LAYERS, LINEAR_ATTN_LAYERS, ATTN_NUM_HEADS
    global ATTN_OUTPUT_GATE, FULL_ATTN_STATES, LINEAR_ATTN_STATES, N_STATES

    N_LAYERS = arch["n_layers"]
    FULL_ATTN_LAYERS = list(arch["full"])
    LINEAR_ATTN_LAYERS = list(arch["linear"])
    ATTN_NUM_HEADS = arch["n_heads"]
    ATTN_OUTPUT_GATE = arch["output_gate"]
    FULL_ATTN_STATES = [L + 1 for L in FULL_ATTN_LAYERS]
    LINEAR_ATTN_STATES = [L + 1 for L in LINEAR_ATTN_LAYERS]
    N_STATES = N_LAYERS + 1


def _is_hybrid() -> bool:
    """True when the model has two attention families to contrast."""
    return bool(LINEAR_ATTN_LAYERS)

METRICS = {"cosine": "cosine", "frobenius": "frobenius",
           "euclidean": "euclidean",
           "cka": "cka", "bw": "bures_wasserstein",
           "mmd": "mmd", "energy": "energy"}
METRIC_COLS = list(METRICS)

#: Metrics that discard each row's magnitude. They are the ones a centering
#: transform interacts with, and not in the direction one might expect: after
#: centering, a row's magnitude *is* its distance from the fleet centroid, so
#: normalizing it away discards exactly what centering exposed. `euclidean` is
#: in the grid to make that visible — it is translation-invariant, so its
#: centered and raw cells are identical by construction, which is the reference
#: the scale-invariant columns should be read against.
SCALE_INVARIANT = ("cosine", "frobenius", "cka")

#: Metrics that read a representation as a *sample* rather than an indexed list.
#: They need more than one row for the same reason CKA does, but for a different
#: reason than BW: BW has a rank-1 covariance on one row (degenerate but
#: defined), whereas a one-point sample has no distribution to estimate.
DISTRIBUTIONAL = ("mmd", "energy")

#: input dim per projection, which is what constrains a BW selection
PROJ_DIN = {"q": 2560, "q_query": 2560, "q_gate": 2560, "k": 2560, "v": 2560,
            "qkv": 2560, "z": 2560, "o": 4096, "out": 4096}

NO_CKA_GROUP = "CKA is single-block:\nno layer grouping"
NO_CKA_ROWS = "CKA needs >1 row"
NO_BW_ROWS = "BW is rank-1 here\n(same as cosine)"
NO_DIST_ROWS = "no distribution\nin a single row"
NO_DIST_STRUCT = "structural has no\nrow sample"
NO_EUCLID_STRUCT = "structural frobenius\nis already unnormalized"


CENTERED_EUCLID = "same as uncentered:\neuclidean is\ntranslation-invariant"


def _redundant(col: str, tf) -> str | None:
    """Why a (metric, transform) cell would duplicate its untransformed twin.

    Both centering modes subtract *one* array from every model, so they are a
    translation of the whole collection, and a translation cannot change a
    Euclidean distance. Computing the cell would produce a numerically identical
    panel under a label implying otherwise, which reads as a coincidence rather
    than an identity. Whitening is a genuine linear map and is exempt.
    """
    from src.analysis.surrogates import transform_key
    if col == "euclidean" and tf is not None and transform_key(tf).startswith("centered"):
        return CENTERED_EUCLID
    return None


def _no_rows(col: str) -> str | None:
    """Why *col* cannot run on a single-row (pooled ``mean``) representation."""
    if col == "cka":
        return NO_CKA_ROWS
    if col == "bw":
        return NO_BW_ROWS
    if col in DISTRIBUTIONAL:
        return NO_DIST_ROWS
    return None


def _blocked(single_row: bool, tf) -> dict[str, str]:
    """``{column: reason}`` for every cell this (representation, transform) lacks."""
    out = {}
    for col in METRIC_COLS:
        reason = _redundant(col, tf)
        if reason is None and single_row:
            reason = _no_rows(col)
        if reason is not None:
            out[col] = reason
    return out


class SuiteCache:
    """Read-through access to ``06_collections`` for the figure suite.

    The suite used to recompute every distance matrix and every MDS embedding on
    every run while the cache that stores exactly those two things sat unused;
    this is the wiring, and ``docs/notes/caching_collections.md`` is why it took
    a note first.

    Two rules make it safe to read back:

    **Row order is not in the handle.** ``collection_key`` sorts the model
    entries before hashing, so a matrix written in ``sort_by_mixture`` order and
    one written in cache-scan order collide. Every load therefore goes through
    :meth:`DistanceMatrix.reindex`, which permutes into the caller's order and
    raises on an id the stored matrix does not hold. Without it the cache would
    make ``docs/notes/row_order_bug.md`` permanent: the same wrong number every
    run, which reads as a result rather than as a defect.

    **Geometries are refitted, not permuted.** A stored embedding is served only
    when its ``model_ids`` match the caller's exactly; otherwise it is refitted.
    Restricting or reordering a fit is not the same operation as permuting a
    symmetric matrix, and an MDS fit is only defined up to rotation anyway.

    The handle each matrix came from is kept in ``_handles``, keyed by ``id()``
    of the matrix and holding a reference to it — the reference is what makes the
    key safe, since a freed object's ``id()`` can be reused by another.
    """

    def __init__(self, cache_root, enabled: bool = True,
                 read: bool = True) -> None:
        """*read* is what ``--no-cache`` turns off, and it turns off **reads
        only**.

        A run that neither reads nor writes cannot be compared against a warm
        one: the cold run would leave an empty cache, the next run would compute
        cold as well, and their digests would match while testing nothing. A
        cold run must therefore populate the cache it is the control for.
        *enabled* is the separate question of whether this object has a cache at
        all, which is how the module-level default stays inert on import.
        """
        self.root = Path(cache_root)
        self.enabled = enabled
        self.read = read
        self._cc = None
        if enabled:
            from src.cache import CollectionCache
            self._cc = CollectionCache(self.root)
        self._handles: dict[int, tuple[object, str]] = {}
        self.hits = 0
        self.misses = 0

    # -- distance matrices --------------------------------------------------

    def distance_matrix(self, compute, *, taxonomy, ids, metric, model_entries,
                        transform=None, surrogate=None, label=None):
        """Read the matrix back if any run has stored it; otherwise *compute* it.

        *surrogate* is the resolved selector dict, never the row's display label:
        ``"late third"`` is editable prose, and redefining which layers it names
        without changing the string would serve a matrix built from the old
        definition. The label is still recorded in ``index.json``, where it makes
        an opaque hash directory identifiable but keys nothing.
        """
        if self._cc is None:
            return compute()

        from src.analysis.comparison import collection_handle

        handle = collection_handle(self._cc, taxonomy, metric, model_entries,
                                   transform=transform, surrogate=surrogate)
        dm = None
        if self.read and self._cc.exists(handle):
            try:
                dm = self._cc.load_distance_matrix(handle).reindex(list(ids))
                self.hits += 1
            except (ValueError, KeyError, OSError) as exc:
                # A stored matrix that cannot serve this request is a miss, not a
                # failure — but say so, because a cache that quietly recomputes
                # forever looks exactly like a cache that is working.
                print(f"      cache: recomputing {handle} — "
                      f"{type(exc).__name__}: {exc}")
                dm = None
        if dm is None:
            self.misses += 1
            dm = compute()
            self._cc.save_distance_matrix(
                dm, handle, model_entries=list(model_entries), label=label,
                config=self._leaf_config(taxonomy, model_entries, transform, surrogate),
            )
        self._handles[id(dm)] = (dm, handle)
        return dm

    @staticmethod
    def _leaf_config(taxonomy, model_entries, transform, surrogate) -> dict:
        """What the leaf ``config.json`` records, so a collection stays traceable.

        The directory name is a digest, so the parts that went into it are
        written out in full: this is the only way back from a stored matrix to
        the selector and the tensors it was built from.
        """
        from src.analysis.surrogates import transform_key
        return {
            "taxonomy": taxonomy,
            "source": "scripts/make_simplex3_figures.py",
            "selectors": dict(surrogate or {}),
            "transform": transform_key(transform),
            "representations": [
                {"model_id": e["model_id"], "artifact_path": e["artifact_path"],
                 "surrogate_hash": e["surrogate_hash"]}
                for e in model_entries
            ],
        }

    # -- geometries ---------------------------------------------------------

    def geometry(self, dm, *, n_components: int, random_state: int):
        """The MDS embedding of *dm*, read back only if it is the same fit.

        Served from disk only when the stored ``model_ids`` match *dm*'s exactly.
        A stored fit under another order is left alone rather than permuted: it
        is valid for the collection it was fitted on, and this caller gets a
        fresh fit instead.
        """
        kwargs = {"random_state": random_state}
        handle = None
        if self._cc is not None:
            entry = self._handles.get(id(dm))
            handle = entry[1] if entry is not None else None

        if handle is not None and self.read:
            try:
                geo = self._cc.load_geometry(handle, "mds", n_components,
                                             mds_kwargs=kwargs)
                if list(geo.model_ids) == list(dm.model_ids):
                    return geo
            except (FileNotFoundError, ValueError, KeyError, OSError):
                pass

        geo = fit_geometry(dm, method="mds", n_components=n_components, **kwargs)
        if handle is not None:
            self._cc.save_geometry(handle, geo, mds_kwargs=kwargs)
        return geo

    def report(self) -> str:
        if not self.enabled:
            return "collection cache: off, everything recomputed"
        if not self.read:
            return (f"collection cache: reads bypassed (--no-cache), "
                    f"{self.misses} recomputed and written under "
                    f"{self.root}/06_collections")
        return (f"collection cache: {self.hits} hit(s), {self.misses} miss(es) "
                f"under {self.root}/06_collections")


#: Set in `main()`. Off until then, so importing this module never writes to a
#: cache and a caller that only wants one function gets no hidden state.
SUITE_CACHE = SuiteCache(CACHE_ROOT, enabled=False)


def metric_row(idx, taxonomy, ids, tf, blocked=None, label=None, **selectors):
    """One grid row: every metric column at a single fixed selector.

    The representations are resolved **once** and every column is computed from
    that one list. Calling ``_compute_distance_matrix`` per column instead
    re-read the same 16 tensors seven times over, which — not any of the
    distances — was what made the suite slow once the grid widened.

    Resolution happens before the cache is consulted, and unconditionally. That
    is the same order ``build_taxonomy_artifacts`` uses and for the same reason:
    a collection is keyed on the artifact each model *resolved to*, which is not
    known until it has been resolved. The cache saves the pairwise computation,
    which is the expensive half — the seven metric columns share one read either
    way.

    *blocked* maps a column to the reason it has no value here, so a structural
    absence is still rendered in place rather than computed and discarded.
    """
    blocked = blocked or {}
    reps, order, model_entries = resolve_ordered(
        idx, taxonomy, ids, transform=tf, with_identity=True, **selectors)
    out = {}
    for col in METRIC_COLS:
        if col in blocked:
            out[col] = blocked[col]
            continue
        out[col] = SUITE_CACHE.distance_matrix(
            lambda col=col: _distances(idx, taxonomy, METRICS[col], ids, reps,
                                       order=order),
            taxonomy=taxonomy, ids=ids, metric=METRICS[col],
            model_entries=model_entries, transform=tf, surrogate=selectors,
            label=label,
        )
    return out


def _bw_mixed(projs) -> str:
    dims = sorted({PROJ_DIN[p] for p in projs})
    return f"BW cannot stack\ninput dims {dims}"


def thirds(xs):
    """Split a layer list into three contiguous, near-equal depth bands."""
    xs = list(xs)
    k = len(xs)
    return xs[: k // 3], xs[k // 3: 2 * k // 3], xs[2 * k // 3:]


# ── Level builders ────────────────────────────────────────────────────────────

def behavioral_cells(idx, ids, surrogates=True):
    def sel(replicates, sampling, **over):
        base = {"draw": DRAW, "max_new_tokens": MAX_NEW_TOKENS,
                "replicates": replicates, "sampling_hash": sampling,
                "embedder_hash": EMBEDDER, "replicate_reduction": "all",
                "view": "matrix", "normalize": "none", "representation": "matrix"}
        return dict(base, **over)

    per_query = {"replicate_reduction": "mean", "representation": "matrix",
                 "renormalize": True}
    model_mean = {"replicate_reduction": "mean", "representation": "mean",
                  "renormalize": True}

    #: (selector, transform) per row.
    rows = {
        # greedy `per query` is omitted, not forgotten: averaging over a single
        # replicate is the identity, so it would duplicate `per generation`.
        "greedy · per generation":  (sel(1, SAMP_GREEDY), None),
        "greedy · model mean":      (sel(1, SAMP_GREEDY, **model_mean), None),
        "R=16 · per generation":    (sel(16, SAMP_SAMPLED), None),
        "R=16 · per query":         (sel(16, SAMP_SAMPLED, **per_query), None),
        "R=16 · model mean":        (sel(16, SAMP_SAMPLED, **model_mean), None),
    }
    if surrogates:
        rows.update({
            # `rowwise` on the R=16 matrix subtracts, from each of a model's 1600
            # rows, the fleet mean of *that* row. Rows are query-major and every
            # model was run on the same draw in the same order, so row k is the
            # same (question, replicate index) slot everywhere — which is what
            # makes the subtraction meaningful. The replicate index is not shared
            # in any deeper sense (they are independent samples), but averaging
            # over the 16 models at a fixed slot still estimates that question's
            # fleet-average answer, which is the nuisance being removed.
            "R=16 · per generation · centered":
                (sel(16, SAMP_SAMPLED), centered("rowwise")),
            "R=16 · per generation · whitened":
                (sel(16, SAMP_SAMPLED), whitened(0.1, "rowwise")),
            # The pooled surrogate has one row per model, so `grand` is the only mode
            # available and it reduces to subtracting the fleet centroid.
            "R=16 · model mean · centered":
                (sel(16, SAMP_SAMPLED, **model_mean), centered("grand")),
        })

    cells = {}
    for row, (selector, tf) in rows.items():
        blocked = _blocked(selector.get("representation") == "mean", tf)
        got = metric_row(idx, "behavioral", ids, tf, blocked, label=row,
                         behavioral_selector=selector)
        cells.update({(row, c): v for c, v in got.items()})
    return list(rows), cells


def _fsel(layers):
    return {"draw": DRAW, "mode": "input", "pooling": "mean", "layers": layers,
            "view": "concat", "normalize": "layer", "max_new_tokens": None}


def functional_layer_rows():
    return {
        # h0 is the negative control: LoRA never touches the embedding matrix, so
        # all 16 models are identical here and whatever a metric reports is its
        # own noise floor.
        "h0 · embeddings (control)":  _fsel([0]),
        "h1 · first linear-attn":     _fsel([1]),
        "h4 · first full-attn":       _fsel([4]),
        "h16 · mid-stack (full-attn)": _fsel([16]),
        "h32 · final hidden state":   _fsel([32]),
    }


def functional_group_rows(surrogates=True):
    early, mid, late = thirds(range(1, N_STATES))
    rows = {
        f"all {N_STATES} layers (reference)": (_fsel(None), None),
        "early third":              (_fsel(list(early)), None),
        "middle third":             (_fsel(list(mid)), None),
        "late third":               (_fsel(list(late)), None),
    }
    if _is_hybrid():
        # Only meaningful when there are two families to tell apart; on a
        # uniform model "full-attn outputs" would just restate the reference row.
        rows["full-attn outputs"] = (_fsel(FULL_ATTN_STATES), None)
        rows["linear-attn outputs"] = (_fsel(LINEAR_ATTN_STATES), None)
    if surrogates:
        # `rowwise`: row i is query i of the shared draw in every model, so this
        # subtracts the base model's own reading of each prompt — which is most
        # of a hidden state, and identical across the 16 by construction since
        # LoRA only perturbs it.
        rows[f"all {N_STATES} layers · centered"] = (_fsel(None), centered("rowwise"))
        rows["late third · centered"] = (_fsel(list(late)), centered("rowwise"))
    return rows


def functional_cells(idx, ids, rows):
    cells = {}
    for row, (selector, tf) in rows.items():
        got = metric_row(idx, "functional", ids, tf, _blocked(False, tf),
                         label=row, functional_selector=selector)
        cells.update({(row, c): v for c, v in got.items()})
    return list(rows), cells


#: `matrix` keeps all 1000 per-document embeddings; `mean` is the (1, 768)
#: centroid that was the only stored surrogate until 2026-08-24.
DATASET_MEAN = {"n_samples": 1000, "seed": 0, "representation": "mean"}
DATASET_MATRIX = {"n_samples": 1000, "seed": 0, "representation": "matrix"}


def dataset_rows(idx, ids, surrogates=True):
    """(selector, transform) per row, dropping surrogates the cache cannot serve.

    The `matrix` surrogate is authored, not derived — see
    ``docs/notes/dataset_embedding_layout.md`` §4 — so it exists only if the
    re-embed job has run. Its rows are omitted with a printed note rather than
    raising, so the no-GPU surrogates still render on a cache that predates it.
    """
    rows = {"dataset text · mean · n1000_s00": (DATASET_MEAN, None)}
    if not surrogates:
        return rows
    # The 16 centroids share one large "Yahoo answer register" direction, which
    # puts every raw cosine distance in 0.00-0.03. Removing it makes the mixture
    # geometry the whole of what is measured rather than a perturbation on it.
    rows["dataset text · mean · centered"] = (DATASET_MEAN, centered("grand"))

    if _dataset_matrix_available(idx, ids):
        # `grand`, not `rowwise`: row i is the i-th sampled document of *this*
        # recipe. Two recipes' row i are unrelated documents, so a per-row fleet
        # mean would be an average over 16 arbitrary texts.
        rows["dataset text · matrix · n1000_s00"] = (DATASET_MATRIX, None)
        rows["dataset text · matrix · centered"] = (DATASET_MATRIX, centered("grand"))
    else:
        print("    no 'matrix' dataset surrogate in the cache — skipping those "
              "rows. Run jobs/simplex3_qwen/02_embed_matrix.sh to author it.")
    return rows


def _dataset_matrix_available(idx, ids) -> bool:
    from src.analysis.comparison import _dataset_embedding_reps
    try:
        _dataset_embedding_reps(idx, DATASET_EMBEDDER, DATASET_MATRIX)
    except Exception:
        return False
    return True


def dataset_cells(idx, ids, rows):
    cells = {}
    for row, (selector, tf) in rows.items():
        blocked = _blocked(selector["representation"] == "mean", tf)
        got = metric_row(idx, "dataset_embedding", ids, tf, blocked, label=row,
                         dataset_selector=selector,
                         embedder_hash=DATASET_EMBEDDER)
        cells.update({(row, c): v for c, v in got.items()})
    return list(rows), cells


# ── Structural ────────────────────────────────────────────────────────────────
#   Built directly on src.notebook.structure, whose four builders are all defined
#   on the LoRA product B @ A and evaluate it through a low-rank identity rather
#   than forming the d×d matrix.

def _structural_dm(weights, names, layers, projs, metric_tag):
    from src.notebook.structure import (
        bures_wasserstein_distance_matrix, cka_distance_matrix,
        cosine_similarity_matrix, frobenius_distance_matrix,
    )
    if metric_tag == "cosine":
        n, S = cosine_similarity_matrix(weights, layers=layers, projections=projs)
        return as_distance_matrix(n, S, "cosine", "structural", similarity=True)
    if metric_tag == "frobenius":
        n, D = frobenius_distance_matrix(weights, layers=layers, projections=projs)
        return as_distance_matrix(n, D, "frobenius", "structural")
    if metric_tag == "bw":
        n, D = bures_wasserstein_distance_matrix(
            weights, layers=layers, projections=projs)
        return as_distance_matrix(n, D, "bures_wasserstein", "structural")
    if metric_tag == "cka":
        if len(layers) != 1 or len(projs) != 1:
            return NO_CKA_GROUP
        n, D = cka_distance_matrix(weights, layers[0], projs[0])
        return as_distance_matrix(n, D, "cka_linear", "structural")
    if metric_tag in DISTRIBUTIONAL:
        # The structural builders work on the LoRA factors and never form the
        # d x d product, so there is no cloud of feature vectors here to read as
        # a sample. The rows of A are a basis, not draws from a distribution.
        return NO_DIST_STRUCT
    if metric_tag == "euclidean":
        # `frobenius_distance_matrix` in src.notebook.structure is already the
        # un-normalized norm of the difference of the two B@A products, so the
        # structural level has only one of the two forms and it is this one.
        return NO_EUCLID_STRUCT
    raise ValueError(metric_tag)


def _load_weights(names, layers, projs):
    from src.notebook.lora_weights import load_lora_weights
    return load_lora_weights(
        names, ADAPTER_ROOT, layer_indices=list(layers), projections=list(projs),
        attn_num_heads=ATTN_NUM_HEADS)


def _structural_grid(idx, names, ids, specs):
    """specs: {row_label: (layers, projections)} -> (rows, cells).

    The weights are read **once** for the union of every row's selection, not once
    per row. Each adapter is ~50 MB and the rows overlap heavily, so per-row
    loading re-read tens of gigabytes and left the job I/O-bound at single-digit
    CPU. The builders already take `layers`/`projections` and intersect against
    what is present, so one collection serves every row.

    That read is now **deferred** until a cell actually misses the cache. Unlike
    the other three levels, structural can be keyed without reading anything: its
    identity is the adapter paths plus the (layers, projections) view, which is
    what `_structural_identity` hashes. So a fully cached grid touches no adapter
    file at all, which is the whole point at ~50 MB apiece.

    The rows are relabelled from bare adapter names to the full model ids
    *before* anything is stored, so a stored structural matrix is addressed in
    the same namespace as every other level's.
    """
    from src.analysis.comparison import _positions_for, _structural_identity

    all_layers = sorted({int(l) for layers, _ in specs.values() for l in layers})
    all_projs = sorted({p for _, projs in specs.values() for p in projs})
    lookup = dict(zip(names, ids))
    order = _positions_for(idx, ids)

    loaded = None

    def weights():
        nonlocal loaded
        if loaded is None:
            print(f"    loading {len(names)} adapters × {len(all_projs)} "
                  "projections once …")
            loaded = _load_weights(names, all_layers, all_projs)
        return loaded

    def entries_for(layers, projs):
        """Model entries in *ids* order, keyed on the view this row reads."""
        identity = _structural_identity(idx, list(layers), list(projs))
        return [{**identity[p], "model_id": mid} for p, mid in zip(order, ids)]

    def compute(row, layers, projs, col):
        dm = _structural_dm(weights(), names, list(layers), list(projs), col)
        # `structure.py` labels its rows with bare adapter names, in the order it
        # was handed them. Relabelling and then reindexing makes that explicit
        # rather than incidental, so a cold matrix and a warm one are row-for-row
        # the same object and their digests in `crosslevel_scores.csv` compare.
        dm.model_ids = [lookup.get(m, m) for m in dm.model_ids]
        return dm.reindex(ids)

    cells = {}
    for row, (layers, projs) in specs.items():
        print(f"    {row}")
        for col in METRIC_COLS:
            if col in DISTRIBUTIONAL:
                cells[(row, col)] = NO_DIST_STRUCT
                continue
            if col == "euclidean":
                cells[(row, col)] = NO_EUCLID_STRUCT
                continue
            if col == "bw" and len({PROJ_DIN[p] for p in projs}) > 1:
                cells[(row, col)] = _bw_mixed(projs)
                continue
            if col == "cka" and (len(layers) != 1 or len(projs) != 1):
                # Decided here rather than inside `_structural_dm`, so that what
                # the cache is handed is always a matrix and never a reason.
                cells[(row, col)] = NO_CKA_GROUP
                continue
            cells[(row, col)] = SUITE_CACHE.distance_matrix(
                lambda row=row, layers=layers, projs=projs, col=col:
                    compute(row, layers, projs, col),
                taxonomy="structural", ids=ids, metric=METRICS[col],
                # No `surrogate=`: for structural the view *is* the surrogate —
                # `_structural_identity` hashes (layers, projections) into
                # `surrogate_hash` — so passing the selectors again would key the
                # same collection twice, once per writer.
                model_entries=entries_for(layers, projs), label=row,
            )
    return list(specs), cells


def structural_layer_specs():
    # Single (layer, projection) cells, so CKA is available on every one and no
    # BW selection can straddle two input dims. o_proj / out_proj are the two
    # families' output projections, which makes them the comparable pair.
    def picks(family):
        """First, middle and last of one attention family."""
        return [family[0], family[len(family) // 2], family[-1]] if family else []

    specs = {}
    prefix = "full-attn · " if _is_hybrid() else ""
    for L in picks(FULL_ATTN_LAYERS):
        specs[f"{prefix}layer {L} · o_proj"] = ([L], ["o"])
    for L in picks(LINEAR_ATTN_LAYERS):
        specs[f"linear-attn · layer {L} · out_proj"] = ([L], ["out"])
    return specs


def structural_group_specs():
    se, sm, sl = thirds(FULL_ATTN_LAYERS)
    if not _is_hybrid():
        # Uniform attention: one family, so the family contrast rows and the
        # linear-attn projections do not exist.  What remains is the same set of
        # questions asked of the only family there is.
        #
        # Note what is deliberately absent: the hybrid branch's "all layers · all
        # projections" and "full-attn · q,k,v,o" ask different questions only
        # because there are two families to span.  With one family they are the
        # same selection, so keeping both would put a literally identical row in
        # every group figure and a second copy of it in the collection cache.
        return {
            "all layers · all projections": (range(N_LAYERS), ["q", "k", "v", "o"]),
            "q,k,v (dim-pure)":             (FULL_ATTN_LAYERS, ["q", "k", "v"]),
            "output projections":           (range(N_LAYERS), ["o"]),
            "early third":                  (se, ["q", "k", "v"]),
            "middle third":                 (sm, ["q", "k", "v"]),
            "late third":                   (sl, ["q", "k", "v"]),
        }

    le, lm, ll = thirds(LINEAR_ATTN_LAYERS)
    return {
        "all layers · all projections": (range(N_LAYERS),
                                         ["q", "k", "v", "o", "qkv", "z", "out"]),
        "full-attn · q,k,v,o":          (FULL_ATTN_LAYERS, ["q", "k", "v", "o"]),
        "linear-attn · qkv,z,out":      (LINEAR_ATTN_LAYERS, ["qkv", "z", "out"]),
        # dim-pure companions, so BW has a value at family resolution too
        "full-attn · q,k,v (d_in 2560)": (FULL_ATTN_LAYERS, ["q", "k", "v"]),
        "linear-attn · qkv,z (d_in 2560)": (LINEAR_ATTN_LAYERS, ["qkv", "z"]),
        "output projections (d_in 4096)": (range(N_LAYERS), ["o", "out"]),
        "full-attn · early third":      (se, ["q", "k", "v"]),
        "full-attn · middle third":     (sm, ["q", "k", "v"]),
        "full-attn · late third":       (sl, ["q", "k", "v"]),
        "linear-attn · early third":    (le, ["qkv", "z"]),
        "linear-attn · middle third":   (lm, ["qkv", "z"]),
        "linear-attn · late third":     (ll, ["qkv", "z"]),
    }


def structural_projection_specs():
    p = "full-attn · " if _is_hybrid() else ""
    specs = {f"{p}q_proj (whole)": (FULL_ATTN_LAYERS, ["q"])}
    if ATTN_OUTPUT_GATE:
        # attn_output_gate fuses a gate into q_proj, so half its rows are not
        # queries at all. The halves are interleaved per head, not stacked.
        # Without the flag there is no split to make and q_proj is all queries.
        specs[f"{p}q_proj query half"] = (FULL_ATTN_LAYERS, ["q_query"])
        specs[f"{p}q_proj gate half"] = (FULL_ATTN_LAYERS, ["q_gate"])
    specs[f"{p}k_proj"] = (FULL_ATTN_LAYERS, ["k"])
    specs[f"{p}v_proj"] = (FULL_ATTN_LAYERS, ["v"])
    specs[f"{p}o_proj"] = (FULL_ATTN_LAYERS, ["o"])
    if _is_hybrid():
        specs["linear-attn · in_proj_qkv"] = (LINEAR_ATTN_LAYERS, ["qkv"])
        specs["linear-attn · in_proj_z"] = (LINEAR_ATTN_LAYERS, ["z"])
        specs["linear-attn · out_proj"] = (LINEAR_ATTN_LAYERS, ["out"])
    return specs


# ── Ground truth, for the layer sweep and the cross-level closer ──────────────

#: The three mixture components, as :mod:`src.analysis.ground_truth` names its
#: simplex vertices. The order fixes which column of the weight array is which
#: vertex, so it must match ``mixture_weights``.
VERTICES = ["g1", "g2", "g3"]


def truth_weights(ids):
    """The ``(n, 3)`` ground-truth mixture array, in *ids* order.

    The only experiment-specific part of the ground truth: these adapters carry
    their recipe in their name, so the weights are parsed rather than read from
    a recipe file. Everything downstream of this array — the simplex embedding,
    its distance matrix, and both scores against it — is
    :mod:`src.analysis.ground_truth`.
    """
    return np.vstack([mixture_weights(m) for m in ids])


def truth_dm(ids):
    """Pairwise distances on the ground-truth simplex, in *ids* order.

    Barycentric coordinates ``W @ simplex_vertices(3)``, not the raw weight
    vectors. Those differ by a constant factor of ``1/√2``, and dCor is
    invariant to a constant rescaling, so this reports the same numbers as the
    ``pdist(W)`` this used to compute — while being the same object
    ``truth_geometry`` embeds, rather than a second convention alongside it.
    """
    return simplex_distance_matrix(truth_weights(ids), list(ids), VERTICES)


def truth_geometry(ids):
    """Where each model sits on the ground-truth simplex — the Procrustes target."""
    return simplex_geometry(truth_weights(ids), list(ids), VERTICES)


# ── Figures ───────────────────────────────────────────────────────────────────

def emit(level, rows, cells, outdir, title):
    dm_grid(cells, rows, METRIC_COLS, f"{title} — distance matrices",
            savepath=outdir / f"fig_{level}_dm_grid.png")
    plt.close("all")
    mds_grid(cells, rows, METRIC_COLS, f"{title} — MDS embeddings",
             savepath=outdir / f"fig_{level}_mds_grid.png")
    plt.close("all")


def emit_detail(level, row, cells, outdir, title):
    """One annotated panel per metric for the level's reference surrogate."""
    for col in METRIC_COLS:
        cell = cells.get((row, col))
        if cell is None or isinstance(cell, str):
            continue
        dm_grid({(row, col): cell}, [row], [col],
                f"{title} · {col}", annot=True, panel_w=6.4, panel_h=6.0,
                savepath=outdir / f"fig_{level}_dm_{col}.png")
        plt.close("all")
        mds_grid({(row, col): cell}, [row], [col], f"{title} · {col}",
                 panel_w=5.4, panel_h=5.0,
                 savepath=outdir / f"fig_{level}_mds_{col}.png")
        plt.close("all")


def layer_sweep(idx, ids, outdir):
    """Where in the stack does mixture identity live? One line per metric."""
    tdm = truth_dm(ids)
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = list(range(N_STATES))

    # Layer-major, not metric-major: one hidden state's tensors are read once and
    # all seven metrics run off them. The other way round re-read the whole stack
    # per metric, which is 7x the I/O for identical numbers.
    scores = {col: [] for col in METRIC_COLS}
    for h in xs:
        selector = _fsel([h])
        reps, order, model_entries = resolve_ordered(
            idx, "functional", ids, functional_selector=selector,
            with_identity=True)
        for col in METRIC_COLS:
            try:
                dm = SUITE_CACHE.distance_matrix(
                    lambda col=col: _distances(idx, "functional", METRICS[col],
                                               ids, reps, order=order),
                    taxonomy="functional", ids=ids, metric=METRICS[col],
                    model_entries=model_entries,
                    surrogate={"functional_selector": selector},
                    label=f"sweep · h{h}",
                )
                scores[col].append(dcor_vs_truth(dm, tdm))
            except Exception as exc:
                # h0 is all-zero by construction, so some metrics legitimately
                # have nothing to report there. Say which layer and why rather
                # than letting a NaN propagate silently into the plot.
                print(f"    {col}: h{h} skipped — {type(exc).__name__}: {exc}")
                scores[col].append(np.nan)

    for col in METRIC_COLS:
        ys = scores[col]
        ax.plot(xs, ys, marker="o", ms=2.6, lw=1.1, label=col)
        if np.isnan(ys).all():
            print(f"    {col}: no finite values across all {len(xs)} layers")
        else:
            print(f"    {col}: best h{int(np.nanargmax(ys))} = {np.nanmax(ys):.3f}")
    # Distinguishable from the whitegrid gridlines, which are also pale grey —
    # at 0.85 these marks were invisible and the legend entry meaningless.
    for h in FULL_ATTN_STATES:
        ax.axvline(h, color="0.62", lw=0.9, ls=(0, (4, 2)), zorder=0)
    ax.axvline(FULL_ATTN_STATES[0], color="0.62", lw=0.9, ls=(0, (4, 2)), zorder=0,
               label="full-attention output")
    ax.set_xlim(-0.5, N_STATES - 0.5)
    ax.set_xlabel("hidden state (h0 = embeddings)")
    ax.set_ylabel("distance correlation vs ground-truth simplex")
    ax.set_title("Functional level — mixture signal by depth", fontsize=10)
    ax.legend(fontsize=6, ncol=4)
    save_figure(fig, str(outdir / "fig_functional_layer_sweep.png"))
    plt.close("all")


#: The MDS seed every cross-level number is computed under. `MDSGeometry`
#: initialises randomly, so this is load-bearing: it is the seed
#: `crosslevel_mds` fits its panels with, and both the Procrustes disparity and
#: the stress reported beside it describe *that* configuration.
MDS_SEED = 0


class RungScore(NamedTuple):
    """One scored (surrogate, metric) cell of one level.

    ``dcor`` runs 0→1 better; ``procrustes`` runs 1→0 better. They are not two
    readings of one quantity: dCor scores the distance matrix and never embeds,
    while the disparity scores the configuration the MDS panel actually draws
    and so inherits the distortion ``stress`` reports.
    """

    dcor: float
    procrustes: float
    stress: float
    row: str
    col: str
    dm: object


def rank_surrogates(level_cells, ids, tdm=None, tgeo=None):
    """Every computed (surrogate, metric) cell of one level, scored against the truth.

    Returns ``[RungScore, ...]`` sorted by **dCor** descending. The sort key is
    deliberately still dCor even though a second score is now reported: the
    winner each figure draws is the dCor winner, and this is the ranking the
    tracked tables record. Scoring every cell rather than a designated reference
    surrogate is the point of the surrogate work: which surrogate recovers the simplex
    best is the question, so it cannot be answered by hardcoding one and
    reporting it.

    *tdm* and *tgeo* are the ground truth in matrix and configuration form; both
    are rebuilt from *ids* when omitted. Pass them when scoring several levels
    against the same models, which is what ``cross_level`` does.
    """
    tdm = truth_dm(ids) if tdm is None else tdm
    tgeo = truth_geometry(ids) if tgeo is None else tgeo
    scored = []
    for (row, col), cell in level_cells.items():
        if cell is None or isinstance(cell, str):
            continue
        if float(np.max(np.abs(cell.matrix))) <= 1e-6:
            continue                     # the h0 control: no geometry to score
        try:
            # One embedding, three uses: the disparity, the stress that
            # qualifies it, and nothing else — fitting it twice under two seeds
            # would let the reported numbers describe different configurations.
            geo = SUITE_CACHE.geometry(cell, n_components=2,
                                       random_state=MDS_SEED)
            scored.append(RungScore(
                dcor=dcor_vs_truth(cell, tdm),
                procrustes=disparity_vs_truth(cell, tgeo, geometry=geo),
                stress=float(kruskal_stress(cell, geo)),
                row=row, col=col, dm=cell,
            ))
        except Exception as exc:
            print(f"    scoring {row!r}/{col} skipped — "
                  f"{type(exc).__name__}: {exc}")
    return sorted(scored, key=lambda s: -s.dcor)


def _bold_if(value, flag) -> str:
    """A 4-decimal markdown cell, bolded when *flag*."""
    return f"**{value:.4f}**" if flag else f"{value:.4f}"


def _matrix_digest(dm) -> str:
    """Content hash of a distance matrix, for telling two runs apart.

    Rounded before hashing so that a re-run differing only in the last bits of a
    float still hashes the same — the question this answers is "is this the same
    matrix", not "is this the same float noise". Truncated to 16 hex characters:
    this identifies matrices within a run, it does not authenticate them.
    """
    m = np.ascontiguousarray(np.round(np.asarray(dm.matrix, dtype=np.float64), 12))
    return hashlib.sha256(m.tobytes()).hexdigest()[:16]


def write_scores_csv(per_level_scores, path) -> None:
    """One row per scored cell, from the same numbers the tables are built from.

    Run output, not cache: nothing keys off it and nothing reads it back, so it
    is safe to delete and safe to change. It exists because the markdown tables
    are for reading and this is for diffing — and because ``matrix_sha256`` is
    the evidence a later change would need to show that a cached distance matrix
    is the matrix a cold run produces (see ``docs/notes/caching_collections.md``).
    """
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["level", "surrogate", "metric", "dcor", "procrustes", "stress",
                    "n_models", "matrix_sha256"])
        for lvl, ranked in per_level_scores.items():
            for s in ranked:
                w.writerow([lvl, s.row, s.col, f"{s.dcor:.6f}",
                            f"{s.procrustes:.6f}", f"{s.stress:.6f}",
                            len(s.dm.model_ids), _matrix_digest(s.dm)])


#: Display name and left-to-right order for the cross-taxonomy figure. The
#: sequence runs from the level furthest from the model's parameters to the level
#: closest to its behaviour: what it was trained on, what its weights became,
#: what its activations do, what it says.
LEVEL_ORDER = [
    ("dataset_embedding", "Dataset"),
    ("structural", "Structural"),
    ("functional", "Functional"),
    ("behavioral", "Behavioral"),
]


def cross_level(per_level, ids, outdir, metric_override=None, suffix=""):
    """Each level's best-scoring surrogate side by side, plus the agreement table.

    *metric_override* maps a level to a metric name that level must be read
    under, e.g. ``{"dataset_embedding": "cosine"}``. The surrogate is still chosen by
    dCor, but only among that metric's cells. It exists to ask what a level looks
    like when it is held to the same metric as the rest of the figure, rather
    than to whichever metric happens to score best on it — a real question here,
    since the dataset level is the only one whose unrestricted winner is not
    `cosine`.

    It is a question and not a correction. Pinning the dataset level to `cosine`
    costs 0.043 dCor and takes that panel's MDS stress from 0.014 to 0.253, so
    the metric consistency is bought at a visible price. Both variants are
    written for that reason; see ``docs/CHANGELOG.md`` and
    ``notebooks/8_crosslevel_dataset_cosine.ipynb`` for the two readings that
    survive it.

    *suffix* is appended to every output filename, so an override variant sits
    beside the unrestricted one rather than overwriting it.
    """
    metric_override = metric_override or {}
    # Scored once per level and reused for the winners, the detail tables and
    # the csv. Ranking each level twice used to recompute every dCor; with a
    # second score and an MDS fit per cell that waste is no longer small.
    tdm, tgeo = truth_dm(ids), truth_geometry(ids)
    per_level_scores = {lvl: rank_surrogates(cells, ids, tdm=tdm, tgeo=tgeo)
                        for lvl, cells in per_level.items()}

    winners, table_rows = {}, []
    for lvl, ranked in per_level_scores.items():
        want = metric_override.get(lvl)
        if want is not None:
            restricted = [s for s in ranked if s.col == want]
            if restricted:
                ranked = restricted
            else:
                # Fall back rather than drop the level: a missing metric should
                # cost the constraint, not the panel.
                print(f"    {lvl}: no {want!r} cell — using its overall best surrogate")
        if not ranked:
            print(f"    {lvl}: no scorable cell — omitted from the comparison")
            continue
        best = ranked[0]
        winners[lvl] = best
        table_rows.append((lvl, best))

    if not winners:
        return

    # LEVEL_ORDER fixes the sequence and the display names; a level it does not
    # name is appended under its raw key, so adding a taxonomy shows up in the
    # figure rather than being silently dropped from it.
    display = dict(LEVEL_ORDER)
    ordered = [(display[lvl], lvl) for lvl, _ in LEVEL_ORDER if lvl in winners]
    ordered += [(lvl, lvl) for lvl in winners if lvl not in display]

    # Name the constraint in the figure itself. A reader comparing the two
    # variants side by side should not have to diff the filenames to find out
    # which panel was pinned to which metric.
    note = ", ".join(f"{display.get(l, l)} pinned to {m}"
                     for l, m in metric_override.items() if l in winners)
    subtitle = "Mixtures from 3 topic groupings from the Yahoo Answers Dataset"
    crosslevel_mds(
        [(name, winners[lvl].dm, winners[lvl].dcor, winners[lvl].procrustes)
         for name, lvl in ordered],
        "Cross-Taxonomy Simplex — MDS",
        subtitle=subtitle + (f" · {note}" if note else ""),
        savepath=outdir / f"fig_crosslevel_mds{suffix}.png",
        random_state=MDS_SEED,
    )
    plt.close("all")
    cells = {("best surrogate", name): winners[lvl].dm for name, lvl in ordered}
    dm_grid(cells, ["best surrogate"], [name for name, _ in ordered],
            "Cross-level comparison — distance matrices, each level's best surrogate"
            + (f" ({note})" if note else ""),
            savepath=outdir / f"fig_crosslevel_dm{suffix}.png")
    plt.close("all")

    # Two scores in one table, running in opposite directions, so the header
    # says which way each one reads rather than leaving it to be inferred.
    header = ["| level | dCor vs ground truth | Procrustes residual (lower=better) "
              "| surrogate | metric |", "|---|---|---|---|---|"]
    best_proc = min((s.procrustes for _, s in table_rows), default=None)
    lines = list(header)
    for lvl, s in table_rows:
        lines.append(f"| {lvl} | {s.dcor:.4f} | {_bold_if(s.procrustes, s.procrustes == best_proc)} "
                     f"| {s.row} | {s.col} |")
    table = "\n".join(lines)

    # The full ranking beside the winners, so a surrogate that wins by a hair does not
    # read as a decisive one, and so a surrogate that helped is visible even when
    # it did not take first place.
    detail = ["", "", "## Every surrogate, per level", ""]
    for lvl, ranked in per_level_scores.items():
        # Rows stay in dCor order — the Procrustes bolding marks the best three
        # of *that* column, which is exactly the interesting case when the two
        # scores disagree about which surrogate recovers the simplex.
        top3 = {id(s) for s in sorted(ranked, key=lambda s: s.procrustes)[:3]}
        detail += [f"### {lvl}", "",
                   "| dCor | Procrustes residual (lower=better) | surrogate | metric |",
                   "|---|---|---|---|"]
        detail += [f"| {s.dcor:.4f} | {_bold_if(s.procrustes, id(s) in top3)} "
                   f"| {s.row} | {s.col} |" for s in ranked]
        detail.append("")

    (outdir / f"crosslevel_agreement{suffix}.md").write_text(
        table + "\n" + "\n".join(detail) + "\n")
    write_scores_csv(per_level_scores, outdir / f"crosslevel_scores{suffix}.csv")
    print(table)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global CACHE_ROOT, ADAPTER_ROOT, SUITE_CACHE
    global BASE_MODEL, BASE_SLUG, DRAW

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
    ap.add_argument("--skip-surrogate", action="store_true",
                    help="omit the centered/whitened surrogates, leaving the raw ones")
    ap.add_argument("--cache-root", default=None,
                    help="the shared cache to read models from and to reuse "
                         f"distance matrices in (default: {CACHE_ROOT})")
    # A warm run reproducing a cold run exactly is the only real test that the
    # reuse is correct, and it cannot be run without a supported way to force the
    # cold one. Compare the `matrix_sha256` column of the two runs'
    # `crosslevel_scores.csv`.
    ap.add_argument("--no-cache", action="store_true",
                    help="recompute everything, ignoring stored results "
                         "(still writes them back)")
    # Run identity.  Architecture is derived from the checkpoint; what cannot be
    # derived is *which run* to plot, so that comes from here.
    ap.add_argument("--base-model", default=BASE_MODEL,
                    help=f"the suite's base model (default: {BASE_MODEL})")
    ap.add_argument("--draw-recipe-hash", default=DRAW["recipe_hash"])
    ap.add_argument("--draw-n", type=int, default=DRAW["n_samples"])
    ap.add_argument("--draw-seed", type=int, default=DRAW["seed"])
    ap.add_argument("--draw-format-id", default=DRAW["prompt_format_id"],
                    help="prompt_format_id of the query draw; '' for a raw suite")
    args = ap.parse_args()

    BASE_MODEL = args.base_model
    BASE_SLUG = BASE_MODEL.replace("/", "--")
    DRAW = {"recipe_hash": args.draw_recipe_hash, "n_samples": args.draw_n,
            "seed": args.draw_seed}
    if args.draw_format_id:
        DRAW["prompt_format_id"] = args.draw_format_id

    arch = architecture(BASE_MODEL)
    apply_architecture(arch)
    print(f"model: {BASE_MODEL}")
    print(f"arch : {N_LAYERS} layers, {ATTN_NUM_HEADS} heads, "
          f"{len(FULL_ATTN_LAYERS)} full-attn / {len(LINEAR_ATTN_LAYERS)} "
          f"linear-attn, output_gate={ATTN_OUTPUT_GATE}")
    surrogates = not args.skip_surrogate

    if args.cache_root:
        CACHE_ROOT = Path(args.cache_root).expanduser().resolve()
        ADAPTER_ROOT = CACHE_ROOT / "03_adapters"
    if not CACHE_ROOT.exists():
        raise SystemExit(
            f"no cache at {CACHE_ROOT}. Pass --cache-root; from a git worktree "
            "the default is derived from this file's location and may not "
            "resolve to the checkout that holds the cache."
        )
    SUITE_CACHE = SuiteCache(CACHE_ROOT, read=not args.no_cache)

    levels = args.level or ["behavioral", "functional", "structural",
                            "dataset_embedding"]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_style("two_col_full")

    idx = scan_cache(str(CACHE_ROOT), base_model_id=BASE_MODEL,
                     behavioral_draw=DRAW, functional_draw=DRAW)
    ids = sort_by_mixture(idx.model_ids)
    names = [Path(m).name for m in ids]
    print(f"cache: {CACHE_ROOT}\nmodels: {len(ids)}")
    print(f"reuse: {'reads bypassed (--no-cache), still writing' if args.no_cache else '06_collections'}")
    if len(ids) != 16:
        raise SystemExit(f"expected 16 models, found {len(ids)} — cache incomplete?")

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ternary_legend(ax, ids, label_models=True)
    ax.set_title("simplex3 mixtures — barycentric colour key", fontsize=9)
    save_figure(fig, str(outdir / "fig_ternary_legend.png"))
    plt.close("all")

    #: level -> every computed cell, for the cross-level ranking at the end
    per_level = {}

    if "behavioral" in levels:
        print("behavioral …")
        rows, cells = behavioral_cells(idx, ids, surrogates)
        emit("behavioral", rows, cells, outdir, "Behavioral level")
        if not args.skip_detail:
            emit_detail("behavioral", rows[0], cells, outdir, "Behavioral")
        per_level["behavioral"] = cells

    if "functional" in levels:
        print("functional (individual layers) …")
        lrows = {k: (v, None) for k, v in functional_layer_rows().items()}
        rows, cells = functional_cells(idx, ids, lrows)
        emit("functional_layers", rows, cells, outdir,
             "Functional level — individual layers")
        print("functional (groupings) …")
        grows, gcells = functional_cells(idx, ids, functional_group_rows(surrogates))
        emit("functional_groups", grows, gcells, outdir,
             "Functional level — layer groupings")
        if not args.skip_detail:
            emit_detail("functional", grows[0], gcells, outdir, "Functional")
        per_level["functional"] = {**cells, **gcells}
        if not args.skip_sweep:
            print("functional layer sweep …")
            layer_sweep(idx, ids, outdir)

    if "structural" in levels:
        structural_cells = {}
        for tag, specs, title in [
            ("structural_layers", structural_layer_specs(),
             "Structural level — individual layers"),
            ("structural_groups", structural_group_specs(),
             "Structural level — layer groupings"),
            ("structural_projections", structural_projection_specs(),
             "Structural level — per projection"),
        ]:
            print(f"{tag} …")
            # `_structural_grid` relabels the bare adapter names structure.py
            # returns onto the full ids the mixture parser and the colour system
            # expect, before storing anything.
            rows, cells = _structural_grid(idx, names, ids, specs)
            emit(tag, rows, cells, outdir, title)
            structural_cells.update(cells)
            if tag == "structural_groups" and not args.skip_detail:
                emit_detail("structural", rows[0], cells, outdir, "Structural")
        per_level["structural"] = structural_cells

    if "dataset_embedding" in levels:
        print("dataset_embedding …")
        rows, cells = dataset_cells(idx, ids, dataset_rows(idx, ids, surrogates))
        emit("dataset_embedding", rows, cells, outdir, "Dataset-embedding level")
        if not args.skip_detail:
            emit_detail("dataset_embedding", rows[0], cells, outdir, "Dataset embedding")
        per_level["dataset_embedding"] = cells

    if len(per_level) > 1:
        print("cross-level …")
        cross_level(per_level, ids, outdir)
        # The same comparison with the dataset level read under cosine instead of
        # its unrestricted frobenius winner — see `cross_level`'s docstring, and
        # `notebooks/8_crosslevel_dataset_cosine.ipynb` for the standalone build.
        print("cross-level (dataset · cosine) …")
        cross_level(per_level, ids, outdir,
                    metric_override={"dataset_embedding": "cosine"},
                    suffix="_dataset_cosine")

    n = len(list(outdir.glob("*.png")))
    print(f"\nwrote {n} figures to {outdir}")
    print(SUITE_CACHE.report())


if __name__ == "__main__":
    main()
