"""Fleet-level transforms applied to representations before they are distanced.

A :class:`~src.core.protocols.DistanceMetric` sees exactly two models, so it can
never subtract a fleet mean or divide by a fleet covariance.  Those are
*collection*-level operations: they are defined by the set of models being
compared, and change when a model joins or leaves it.  This module is where they
live, between :func:`~src.analysis.comparison._resolve_representations` and
:func:`~src.analysis.comparison._distances`.

Why the levels need this
------------------------
Every representation here carries a large component that is shared by every
model in the collection and so encodes nothing about what distinguishes them:

* **dataset** — a recipe's centroid is dominated by "this is Yahoo answer text".
  Cosine distances across the 16 simplex3 recipes span 0.00–0.03, so the mixture
  geometry is a small perturbation riding on one shared direction.
* **behavioral** — all 16 models answer the same 100 questions, so most of a
  generation embedding is question content, identical across models by design.
* **functional** — hidden states are dominated by the base model's own
  representation of the prompt; LoRA moves them a little.

Centering removes that component.  What is left is the between-model variation,
which is the thing a taxonomy is trying to measure.  This changes what cosine
*means* — on centered rows it is a correlation rather than an angle to the
origin — which is why it is an explicit, recorded transform rather than a
silent default.

Every function returns fresh :class:`~src.core.representation.ModelRepresentation`
objects and records what it did in ``metadata["surrogate_transform"]``, so a
figure panel can always say which surrogate it is showing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Literal, Sequence

import numpy as np

from src.core.representation import ModelRepresentation

CenterMode = Literal["grand", "rowwise"]

Transform = Callable[[Sequence[ModelRepresentation]], list[ModelRepresentation]]


def _tagged(
    rep: ModelRepresentation, matrix: np.ndarray, tag: dict
) -> ModelRepresentation:
    """A copy of *rep* carrying *matrix* and a record of the transform applied.

    Transforms compose, so the record is a *list*: centering then whitening must
    not present itself as whitening alone.
    """
    meta = dict(rep.metadata or {})
    meta["surrogate_transform"] = list(meta.get("surrogate_transform", [])) + [tag]
    return replace(rep, matrix=np.asarray(matrix, dtype=np.float32), metadata=meta)


def _check_rowwise(reps: Sequence[ModelRepresentation]) -> int:
    counts = {r.n_queries for r in reps}
    if len(counts) != 1:
        raise ValueError(
            f"mode='rowwise' subtracts the fleet mean of row i from row i of "
            f"every model, which is only defined when they are the same row. "
            f"These representations have row counts {sorted(counts)}. Use "
            f"mode='grand', or reduce to a shared query set first."
        )
    return counts.pop()


def center_representations(
    reps: Sequence[ModelRepresentation],
    mode: CenterMode = "grand",
) -> list[ModelRepresentation]:
    """Subtract the fleet mean from every representation.

    Parameters
    ----------
    mode:
        ``"grand"`` subtracts a single ``(1, d)`` vector — the mean over every
        row of every model.  It makes no assumption that row *i* means the same
        thing in two models, so it is the only mode available for a ``(1, d)``
        pooled representation, and the safe one whenever row counts differ.

        ``"rowwise"`` subtracts the per-row fleet mean: row *i* of the output is
        row *i* minus the average of row *i* across all models.  This is the
        stronger transform where it applies, because it removes the *query's*
        contribution rather than a global average of it — for the behavioral
        level that is the question text, which is identical across models by
        construction and is exactly what should not count as similarity.  It
        requires every model to have the same number of rows in the same order.

    A model's row count is preserved, so the result stays usable with the
    row-aligned metrics (Frobenius, CKA) as well as the permutation-invariant
    ones.
    """
    if not reps:
        return []
    if mode not in ("grand", "rowwise"):
        raise ValueError(f"mode must be 'grand' or 'rowwise', got {mode!r}")

    dims = {r.embedding_dim for r in reps}
    if len(dims) != 1:
        raise ValueError(
            f"centering needs one shared feature dimension, got {sorted(dims)}."
        )

    if mode == "grand":
        # Weighted by rows, not by model: the mean over the pooled row set.  An
        # unweighted average of per-model means would be a different vector
        # whenever row counts differ, and would silently give a model with 100
        # rows the same say as one with 1600.
        total = sum(r.matrix.astype(np.float64).sum(axis=0) for r in reps)
        n_rows = sum(r.n_queries for r in reps)
        mu = (total / n_rows)[None, :]
        return [
            _tagged(r, r.matrix.astype(np.float64) - mu,
                    {"kind": "center", "mode": "grand", "n_models": len(reps)})
            for r in reps
        ]

    _check_rowwise(reps)
    stack = np.stack([r.matrix.astype(np.float64) for r in reps])   # (m, n, d)
    mu = stack.mean(axis=0)                                          # (n, d)
    return [
        _tagged(r, m - mu,
                {"kind": "center", "mode": "rowwise", "n_models": len(reps)})
        for r, m in zip(reps, stack)
    ]


def whiten_representations(
    reps: Sequence[ModelRepresentation],
    shrinkage: float = 0.1,
    mode: CenterMode = "grand",
) -> list[ModelRepresentation]:
    """Center, then decorrelate against the pooled fleet covariance.

    Applies ``X → (X - mu) S^{-1/2}`` where ``S`` is the covariance of the pooled
    rows, shrunk toward a scaled identity::

        S_shrunk = (1 - a) S + a * (tr S / d) I

    *shrinkage* ``a`` is not optional and defaults to a deliberately
    non-negligible 0.1.  With 16 models at d=768 the pooled covariance is badly
    conditioned in exactly the low-variance directions, and ``S^{-1/2}`` without
    shrinkage multiplies those directions — which are estimation noise — by the
    largest factors in the whole map.  The result looks like structure and is
    not.  ``a=0`` is accepted but will raise if the covariance is singular.

    Whitening equalizes variance across directions, so a small consistent
    between-model difference counts as much as a large one.  That is the point
    when the large directions are shared nuisance, and a liability when they are
    real signal — so this is offered alongside :func:`center_representations`
    rather than instead of it.
    """
    if not reps:
        return []
    if not 0.0 <= shrinkage < 1.0:
        raise ValueError(f"shrinkage must be in [0, 1), got {shrinkage}")

    centered = center_representations(reps, mode=mode)
    pooled = np.vstack([r.matrix.astype(np.float64) for r in centered])
    n, d = pooled.shape
    if n <= 1:
        raise ValueError(
            f"whitening needs more than one pooled row to estimate a covariance, "
            f"got {n}. A collection of (1, d) means has {len(reps)} pooled rows; "
            f"center it instead."
        )

    S = (pooled.T @ pooled) / (n - 1)
    if shrinkage > 0:
        S = (1.0 - shrinkage) * S + shrinkage * (np.trace(S) / d) * np.eye(d)

    # Symmetric eigendecomposition rather than an explicit inverse: S is
    # symmetric PSD by construction, and this gives S^{-1/2} directly along with
    # the eigenvalues needed to say *why* it failed when it does.
    evals, evecs = np.linalg.eigh(S)
    if evals.min() <= 0:
        raise np.linalg.LinAlgError(
            f"pooled covariance is singular (min eigenvalue {evals.min():.3e}) "
            f"at shrinkage={shrinkage}. Raise shrinkage, or reduce the feature "
            f"dimension ({d}) below the pooled row count ({n})."
        )
    W = evecs @ np.diag(evals ** -0.5) @ evecs.T

    tag = {"kind": "whiten", "shrinkage": shrinkage, "mode": mode,
           "n_models": len(reps), "condition_number": float(evals.max() / evals.min())}
    return [_tagged(r, r.matrix.astype(np.float64) @ W, tag) for r in centered]


def centered(mode: CenterMode = "grand") -> Transform:
    """``center_representations`` bound to a mode, for passing as ``transform=``."""
    def _t(reps):
        return center_representations(reps, mode=mode)
    _t.transform_key = f"centered_{mode}"                    # type: ignore[attr-defined]
    return _t


def whitened(shrinkage: float = 0.1, mode: CenterMode = "grand") -> Transform:
    """``whiten_representations`` bound to its arguments, for ``transform=``."""
    def _t(reps):
        return whiten_representations(reps, shrinkage=shrinkage, mode=mode)
    _t.transform_key = f"whitened_{mode}_s{shrinkage:g}"      # type: ignore[attr-defined]
    return _t


def transform_key(transform: Transform | None) -> str:
    """A short, stable name for *transform*, for figure labels and cache keys.

    The bound helpers above carry their own name, so the key is read off the
    callable rather than derived by running it — a transform must not have to be
    executed just to be *named*, since a cache lookup needs the name before it
    knows whether the work is worth doing.
    """
    if transform is None:
        return "raw"
    key = getattr(transform, "transform_key", None)
    if key is not None:
        return str(key)
    # A caller's own callable: fall back to its name, which at least changes when
    # the function does.  Anonymous ones collapse to "custom", which is why a
    # transform destined for the collection cache should come from a helper.
    name = getattr(transform, "__name__", "custom")
    return "custom" if name == "<lambda>" else name
