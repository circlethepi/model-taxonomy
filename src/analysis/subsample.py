"""Subsampling a cached probe — fewer queries, fewer replicates — without re-running it.

Every extraction in this project writes its *full* probe to the shared cache: the
behavioral level stores a ``(n_queries · replicates, d)`` matrix and the functional
level a ``(n_queries, L·d)`` one.  A question like "how many queries do you
actually need?" is then a question about **subsets of rows that are already on
disk**, not about new inference.  This module is the machinery for asking it.

Three things live here, and they are separable:

:class:`QueryPool`
    The query draw's own labels.  The sampled-row cache stores *source indices* —
    ``(entry, row)`` pairs — so which mixture component each query came from is
    recorded, and for a class-aware recipe the upstream class label is
    recoverable too.  That is what makes a *composed* subsample possible rather
    than only a uniform one: you can ask for 60 queries that are half group 1, or
    for an even split over the ten yahoo topics.

:func:`draw_replicates`
    The replicate axis, which only the behavioral level has.

:class:`CosineParts`
    The reason this is cheap.  See below.

Why a subset's distance matrix costs nothing
--------------------------------------------
:class:`~src.metrics.vector.CosineDistanceMetric` **flattens** both matrices
before comparing them, so the distance between two models read over a query
subset ``S`` is::

    cos(i, j) = Σ_{q∈S} ⟨v_iq, v_jq⟩ / sqrt( Σ_{q∈S} ‖v_iq‖² · Σ_{q∈S} ‖v_jq‖² )

Every term is a **per-query** scalar.  So one pass over the representations
yields ``num[i,j,q] = ⟨v_iq, v_jq⟩`` and ``sq[i,q] = ‖v_iq‖²``, after which *any*
query subset's full ``(M, M)`` distance matrix is two masked sums — microseconds,
for a ``(16, 16, 1000)`` array that is 2 MB.

This is **exact**, not an approximation: it is the same arithmetic
``CosineDistanceMetric`` performs, rearranged.  A sweep over ten query sizes ×
ten draws therefore costs one pass, not a hundred.  The replicate axis does not
factor the same way — averaging a different set of replicates changes ``v_iq``
itself — so the parts must be rebuilt per replicate draw, which is what
:func:`reduce_replicates` is for.

What this does *not* do
-----------------------
It does not re-derive the pipeline's views.  Load a representation through
:class:`~src.cache.generated_text_cache.GeneratedTextCache` or
:class:`~src.cache.activation_cache.ActivationCache` exactly as
:mod:`src.analysis.comparison` would, hand the result here, and the numbers agree
with the ones a full-probe run reports.  The one deliberate exception is the
behavioral replicate mean, which this module performs itself — it has to, since
choosing *which* replicates is the whole point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.protocols import ModelID

__all__ = [
    "QueryPool",
    "draw_replicates",
    "reduce_replicates",
    "CosineParts",
    "cosine_parts",
    "allocate",
]


# ── composition: how many rows each label contributes ─────────────────────────

def allocate(n: int, labels: np.ndarray, weights: Mapping[int, float] | None) -> dict[int, int]:
    """How many of *n* rows each label gets, by largest remainder.

    *weights* maps a label to its share; ``None`` means "balanced", i.e. equal
    shares over the labels present.  Shares are normalised, so ``{0: 1, 1: 1}``
    and ``{0: 50, 1: 50}`` are the same request.

    **Largest remainder, not rounding**, and for the same reason
    :class:`~src.datasets.class_recipe.ClassDatasetEntry` uses it: independent
    rounding does not sum to *n*, and a subsample that is one row short of what
    was asked for is a silent bias rather than an error.  The realized
    composition can still differ from the requested one when ``n`` is small — at
    ``n=2`` over three groups, one group gets nothing — which is a fact about the
    request, not a defect here, and is why
    :meth:`QueryPool.draw_queries` reports what it actually allocated.

    Raises when a label is asked for more rows than the pool holds: drawing
    without replacement cannot satisfy it, and quietly returning fewer rows would
    make ``n`` mean something different at different sizes.
    """
    present = sorted({int(v) for v in np.unique(labels)})
    if weights is None:
        weights = {ell: 1.0 for ell in present}

    unknown = sorted(set(weights) - set(present))
    if unknown:
        raise ValueError(
            f"composition names label(s) {unknown} that the pool does not hold "
            f"(present: {present})"
        )
    wanted = {int(k): float(v) for k, v in weights.items() if float(v) > 0.0}
    if not wanted:
        raise ValueError("composition has no positive weight")
    total = sum(wanted.values())

    exact = {ell: n * w / total for ell, w in wanted.items()}
    alloc = {ell: int(np.floor(x)) for ell, x in exact.items()}
    short = n - sum(alloc.values())
    # Ties broken by label, so the allocation is a pure function of the request
    # and two runs of the same sweep agree row for row.
    order = sorted(wanted, key=lambda ell: (-(exact[ell] - alloc[ell]), ell))
    for ell in order[:short]:
        alloc[ell] += 1

    for ell, k in alloc.items():
        have = int((labels == ell).sum())
        if k > have:
            raise ValueError(
                f"composition asks for {k} rows of label {ell} but the pool holds "
                f"{have}. Lower n, or widen the composition."
            )
    return alloc


# ── the query draw and its labels ─────────────────────────────────────────────

@dataclass(frozen=True)
class QueryPool:
    """One cached query draw, with whatever labels it can be composed along.

    ``groups`` is always available: it is the *mixture component* each query was
    drawn from — entry 0, 1, 2 … of the recipe — which for the simplex trees is
    exactly the g1/g2/g3 the ground truth is built on.  ``topics`` is the upstream
    class label (for yahoo, the topic 0–9) and is ``None`` until
    :meth:`with_topics` resolves it, because that needs the source dataset while
    everything else here needs only the cache.

    The distinction matters: a group is a *set* of topics under these recipes
    (g1 = topics {0,6,7,9}, and so on), so composing by topic is strictly finer
    than composing by group.
    """

    recipe_hash: str
    n_samples: int
    seed: int
    source_indices: np.ndarray          # (n, 2) of (entry, row)
    groups: np.ndarray                  # (n,) entry index
    sources: list[dict]                 # the draw manifest's source descriptors
    topics: np.ndarray | None = None    # (n,) upstream class label, if resolved

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_draw(cls, cache_root: Path | str, draw: Mapping) -> "QueryPool":
        """Read the draw manifest ``01_datasets/{recipe_hash}/n{n}_s{seed}.json``.

        *draw* is the ordinary draw key — ``{recipe_hash, n_samples, seed, ...}``
        — the same dict a behavioral or functional selector carries, so a caller
        already holding one does not have to take it apart.  Extra keys such as
        ``prompt_format_id`` are ignored: the row draw does not depend on how a
        row is later rendered into a prompt.
        """
        rhash = draw["recipe_hash"]
        n, seed = int(draw["n_samples"]), int(draw["seed"])
        path = Path(cache_root) / "01_datasets" / rhash / f"n{n}_s{seed:02d}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"no draw manifest at {path}. The build step writes it; run "
                f"`run_experiment.py ... --steps build` for this recipe first."
            )
        man = json.loads(path.read_text())
        idx = np.asarray(man["indices"], dtype=np.int64)
        if idx.shape != (n, 2):
            raise ValueError(
                f"{path} holds {idx.shape} source indices, expected ({n}, 2)"
            )
        return cls(
            recipe_hash=rhash,
            n_samples=n,
            seed=seed,
            source_indices=idx,
            groups=idx[:, 0].copy(),
            sources=list(man.get("sources", [])),
        )

    def with_topics(self, class_field: str = "topic", token: str | None = None) -> "QueryPool":
        """This pool with ``topics`` resolved from the upstream dataset.

        Goes through :mod:`src.datasets.source_registry` rather than calling
        ``load_dataset`` here, so the revision recorded in the draw manifest is
        the one that gets loaded — the manifest stores row *positions*, and a
        different revision would silently relabel them.

        Costs one load of the source split (1.4M rows for yahoo, memoised for the
        life of the process) and is therefore opt-in: nothing that composes by
        group needs it.
        """
        from src.datasets import source_registry

        topics = np.empty(len(self.groups), dtype=np.int64)
        for entry, src in enumerate(self.sources):
            rows = self.source_indices[:, 0] == entry
            if not rows.any():
                continue
            ds = source_registry.get(
                src["dataset_id"], src.get("subset"), src.get("split", "train"),
                revision=src.get("revision"), token=token,
            )
            want = self.source_indices[rows, 1]
            col = np.asarray(ds.select(want.tolist())[class_field], dtype=np.int64)
            topics[rows] = col
        return QueryPool(
            recipe_hash=self.recipe_hash, n_samples=self.n_samples, seed=self.seed,
            source_indices=self.source_indices, groups=self.groups,
            sources=self.sources, topics=topics,
        )

    # -- drawing --------------------------------------------------------------

    @property
    def n(self) -> int:
        return int(len(self.groups))

    def labels(self, by: str | None) -> np.ndarray | None:
        """The label array named by *by* — ``"group"``, ``"topic"`` or ``None``."""
        if by is None:
            return None
        if by == "group":
            return self.groups
        if by == "topic":
            if self.topics is None:
                raise ValueError(
                    "this pool has no topic labels; call with_topics() first"
                )
            return self.topics
        raise ValueError(f"unknown label axis {by!r}; expected 'group', 'topic' or None")

    def draw_queries(
        self,
        n: int,
        n_draws: int,
        rng: np.random.Generator,
        *,
        by: str | None = None,
        composition: Mapping[int, float] | None = None,
    ) -> list[np.ndarray]:
        """*n_draws* subsets of size *n*, each drawn **without replacement**.

        With *by* left ``None`` this is simple random sampling from the whole
        pool.  That is the honest default for "how many queries do I need?": the
        composition of a draw then varies as it would in practice, and that
        variation is part of what a band over the draws is measuring.

        With *by* naming a label axis, *composition* gives each label's share
        (``None`` meaning equal shares) and the subset is stratified to match —
        composition is then held fixed and the band shows only *which* queries
        were picked.  See :func:`allocate` for how the shares become counts.

        Returns index arrays into the pool, each sorted ascending.  Sorting is
        not cosmetic: it makes a subset a canonical object, so two draws that
        picked the same queries in a different order are recognisably the same
        draw, and it keeps the downstream gathers sequential.

        When ``n`` equals the pool size there is exactly one subset without
        replacement, and asking for several would return ``n_draws`` copies of
        it.  That is silently allowed — it is what a sweep's top end *is* — but
        callers that draw bands should collapse it rather than plot a zero-width
        band as though it had been measured.
        """
        if not 1 <= n <= self.n:
            raise ValueError(f"n={n} outside 1..{self.n}")

        if by is None:
            if composition is not None:
                raise ValueError("composition needs a label axis; pass by='group' or 'topic'")
            return [np.sort(rng.choice(self.n, size=n, replace=False))
                    for _ in range(n_draws)]

        lab = self.labels(by)
        alloc = allocate(n, lab, composition)
        pools = {ell: np.flatnonzero(lab == ell) for ell in alloc}
        out = []
        for _ in range(n_draws):
            picked = [rng.choice(pools[ell], size=k, replace=False)
                      for ell, k in alloc.items() if k]
            out.append(np.sort(np.concatenate(picked)))
        return out

    def composition_of(self, idx: np.ndarray, by: str = "group") -> dict[int, int]:
        """The realized label counts of one subset — what was drawn, not what was asked."""
        lab = self.labels(by)
        sub = lab[idx]
        return {int(v): int((sub == v).sum()) for v in sorted(np.unique(lab))}


# ── the replicate axis ────────────────────────────────────────────────────────

def draw_replicates(
    n_total: int, r: int, n_draws: int, rng: np.random.Generator
) -> list[np.ndarray]:
    """*n_draws* subsets of *r* replicates out of *n_total*, without replacement.

    Sorted, for the same reason query subsets are.  At ``r == n_total`` there is
    one subset and the caller gets ``n_draws`` copies of it — the replicate
    counterpart of the note on :meth:`QueryPool.draw_queries`.
    """
    if not 1 <= r <= n_total:
        raise ValueError(f"r={r} outside 1..{n_total}")
    return [np.sort(rng.choice(n_total, size=r, replace=False)) for _ in range(n_draws)]


def reduce_replicates(
    X: np.ndarray, rep_idx: np.ndarray | None = None, *, renormalize: bool = True
) -> np.ndarray:
    """Average a replicate subset of *X* and, by default, re-normalize the rows.

    *X* is ``(M, Q, R, d)`` — models, queries, replicates, features — which is the
    stored ``(Q·R, d)`` behavioral matrix reshaped.  **That reshape is only valid
    because storage is query-major**, rows ``q0r0 … q0r(R-1)`` being adjacent;
    :meth:`GeneratedTextCache._build_view` relies on the same fact and the two
    must not disagree.

    The default reproduces the canonical behavioral perspective —
    ``replicate_reduction="mean"`` followed by ``renormalize=True`` — and the
    arithmetic is done in float64 throughout, where the cache round-trips through
    float32 between the two steps.  The difference is far below what a Procrustes
    disparity resolves, but it is a difference, so it is stated rather than
    implied.

    A zero row is left at zero rather than given an arbitrary direction, matching
    :func:`src.analysis.comparison._behavioral_view`.
    """
    V = X if rep_idx is None else X[:, :, rep_idx, :]
    V = V.mean(axis=2, dtype=np.float64)
    if renormalize:
        norms = np.linalg.norm(V, axis=-1, keepdims=True)
        V = np.divide(V, norms, out=np.zeros_like(V), where=norms > 1e-10)
    return V


# ── the per-query decomposition of cosine distance ────────────────────────────

@dataclass(frozen=True)
class CosineParts:
    """Per-query inner products and squared norms for a fleet of models.

    ``num[i, j, q] = ⟨v_iq, v_jq⟩`` and ``sq[i, q] = ‖v_iq‖²``.  Together these
    reconstruct :class:`~src.metrics.vector.CosineDistanceMetric` over *any*
    query subset — see the module docstring — which is what makes a query sweep
    a sequence of masked sums rather than a sequence of re-reads.
    """

    num: np.ndarray                 # (M, M, Q)
    sq: np.ndarray                  # (M, Q)
    model_ids: list[ModelID]

    def distance_matrix(
        self,
        idx: np.ndarray | None = None,
        *,
        taxonomy: str = "behavioral",
        metric: str = "cosine",
    ) -> DistanceMatrix:
        """The ``(M, M)`` cosine distance matrix read over query subset *idx*.

        ``idx=None`` uses every query, which reproduces the full-probe matrix.

        A model whose rows are all zero over the subset has no direction, so its
        distance is 1.0 to everything — the convention
        :class:`CosineDistanceMetric` already uses for a degenerate vector, kept
        here so a subset that happens to select only empty generations behaves
        the same way it would at full size.
        """
        num = self.num if idx is None else self.num[:, :, idx]
        sq = self.sq if idx is None else self.sq[:, idx]
        cross = num.sum(axis=2)
        norm = np.sqrt(sq.sum(axis=1))
        denom = np.outer(norm, norm)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos = np.where(denom > 1e-20, cross / denom, 0.0)
        D = 1.0 - np.clip(cos, -1.0, 1.0)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        return DistanceMatrix(
            matrix=D, model_ids=list(self.model_ids), metric=metric, taxonomy=taxonomy,
        )


def cosine_parts(V: np.ndarray, model_ids: Sequence[ModelID]) -> CosineParts:
    """Decompose ``(M, Q, d)`` per-query representations into :class:`CosineParts`.

    The contraction is written as a reshaped ``@`` rather than an ``einsum`` over
    ``q``: it is the same ``(M·Q, d)`` GEMM BLAS is good at, done ``Q`` times, and
    at the sizes here (16 models, 1000 queries, 768 features) that is the
    difference between a second and a minute.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    M, Q, _ = V.shape
    # (Q, M, d) @ (Q, d, M) -> (Q, M, M), one batched GEMM, then to (M, M, Q).
    W = np.transpose(V, (1, 0, 2))
    num = np.transpose(W @ np.transpose(W, (0, 2, 1)), (1, 2, 0))
    sq = np.einsum("mqd,mqd->mq", V, V)
    return CosineParts(num=np.ascontiguousarray(num), sq=sq, model_ids=list(model_ids))
