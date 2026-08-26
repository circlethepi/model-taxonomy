"""The simplex a dataset recipe implies — the one quantity in this package that is
known *before* any model is trained.

A recipe mixes ``k`` components in known proportions.  Those proportions are
barycentric coordinates by construction: they are non-negative and sum to one, so
every recipe names a point on the ``(k-1)``-simplex whose vertices are the pure
components.  A recipe that draws only from component ``j`` sits *at* vertex ``j``;
a 25/75 mixture sits a quarter of the way along the edge between two vertices.

This is the ground truth :mod:`src.analysis.simplex` was built to be checked
against.  ``anchor_weight_vs_truth`` already accepts a ``(n, k)`` array of known
values but leaves the caller to produce it; everything here produces it, from the
recipe files the pipeline already writes.

What counts as a component
--------------------------
One vertex per *mixture component*, which is finer than one vertex per dataset:
a :class:`~src.datasets.class_recipe.ClassDatasetEntry` that draws from two
classes of one dataset is mixing two things, not one.  So a component is a
``(dataset entry, class value)`` pair when the entry is class-aware, and the bare
entry otherwise, carrying weight ``entry_weight × class_weight``.  For a recipe
that mixes three whole datasets this reduces to exactly three vertices at
``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)``, which is the intuition; for the
class-mixture recipes in the cache it gives the two topics rather than collapsing
them to a single degenerate vertex.

Dimension
---------
``k`` components span a ``(k-1)``-dimensional simplex, and
:func:`~src.analysis.simplex.barycentric` needs ``k`` affinely independent
anchors — so an embedding must have at least ``k-1`` dimensions before the
projection means anything.  It raises otherwise.  Fit with
``fit_geometry(dm, "mds", n_components=len(vertices) - 1)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult
from src.core.protocols import ModelID

__all__ = [
    "MixtureComponent",
    "mixture_weights",
    "ground_truth_weights",
    "truth_matrix",
    "simplex_vertices",
    "simplex_geometry",
    "simplex_distance_matrix",
    "dcor_vs_truth",
    "disparity_vs_truth",
    "pure_anchors",
    "evaluation_points",
]

# Weights are compared against 1.0 after normalisation, which the recipe classes
# already performed in float; this is loose enough for that round trip and tight
# enough to catch a genuinely malformed recipe.
_SUM_TOL = 1e-6


# ── one component of a mixture ─────────────────────────────────────────────────

@dataclass(frozen=True)
class MixtureComponent:
    """One vertex of the ground-truth simplex.

    ``class_value`` is ``None`` for a plain :class:`~src.datasets.recipe.DatasetEntry`,
    and the class label for a class-aware entry.  It is stored as a string
    because ``recipe.json`` round-trips class keys through JSON, where they
    become strings, while the in-memory recipe objects keep them as ints — the
    two must produce the same vertex key or a collection assembled from both
    sources would split one vertex in two.
    """

    dataset_id: str
    split: str = "train"
    subset: str | None = None
    class_field: str | None = None
    class_value: str | None = None

    @property
    def key(self) -> str:
        """Readable, stable identifier used as the vertex label.

        Split and subset appear only when they are not the default, so the
        common case reads as ``yahoo_answers_topics[topic=0]`` rather than
        carrying fields that never vary.
        """
        base = self.dataset_id
        if self.subset:
            base = f"{base}:{self.subset}"
        if self.split and self.split != "train":
            base = f"{base}@{self.split}"
        if self.class_value is not None:
            base = f"{base}[{self.class_field}={self.class_value}]"
        return base


# ── recipe → weights ───────────────────────────────────────────────────────────

def mixture_weights(recipe: Any) -> dict[str, float]:
    """Component key → weight for one recipe.  Weights sum to 1.

    Accepts a :class:`~src.datasets.recipe.DatasetRecipe`, a
    :class:`~src.datasets.class_recipe.ClassAwareDatasetRecipe`, or the
    ``to_dict()`` form — which is what ``recipe.json`` holds, and therefore what
    comes back out of :class:`~src.cache.dataset_embedding_cache.DatasetEmbeddingCache`.
    All three appear in practice, so all three are handled here rather than at
    every call site.

    An entry that names a ``class_field`` but pins neither ``class_filter`` nor
    ``class_weights`` contributes a **single** component for the dataset as a
    whole.  Its weight is known — it is the entry weight — even though the split
    *within* it is only decided when the data loads, and that split does not
    matter while the dataset is one vertex.  See
    :func:`ground_truth_weights` for the one case where it does.
    """
    entries = _normalized_entries(recipe)

    weights: dict[str, float] = {}
    for component, weight in entries:
        # Two entries can legitimately name the same component — the same dataset
        # listed twice, or a class appearing under two entries — so accumulate.
        weights[component.key] = weights.get(component.key, 0.0) + weight

    total = sum(weights.values())
    if not weights or abs(total - 1.0) > _SUM_TOL:
        raise ValueError(
            f"recipe weights sum to {total!r}, expected 1.0. Components: "
            f"{sorted(weights)}"
        )
    return weights


def _normalized_entries(recipe: Any) -> list[tuple[MixtureComponent, float]]:
    """Flatten any recipe form into ``(component, weight)`` pairs."""
    if isinstance(recipe, Mapping):
        entry_dicts = list(recipe.get("datasets", []))
        entry_weights = list(recipe.get("normalized_weights", []))
        class_aware = recipe.get("recipe_type") == "class_aware"
        name = recipe.get("name", "<dict>")
    else:
        entry_dicts = [e.to_dict() for e in getattr(recipe, "datasets", [])]
        entry_weights = list(getattr(recipe, "normalized_weights", []))
        # A ClassAwareDatasetRecipe holds ClassDatasetEntry objects, whose
        # to_dict() emits class_field; a plain DatasetRecipe's does not.
        class_aware = any("class_field" in d for d in entry_dicts)
        name = getattr(recipe, "name", type(recipe).__name__)

    if not entry_dicts:
        raise ValueError(f"recipe {name!r} has no dataset entries")
    if len(entry_weights) != len(entry_dicts):
        raise ValueError(
            f"recipe {name!r} has {len(entry_dicts)} entries but "
            f"{len(entry_weights)} normalized_weights"
        )

    out: list[tuple[MixtureComponent, float]] = []
    for entry, entry_weight in zip(entry_dicts, entry_weights):
        if entry_weight <= 0.0:
            continue  # contributes nothing, and would add a zero-weight vertex

        dataset_id = entry["dataset_id"]
        split = entry.get("split", "train")
        subset = entry.get("subset")
        class_field = entry.get("class_field")
        class_weights = entry.get("normalized_class_weights")

        if class_weights:
            for raw_value, class_weight in class_weights.items():
                if class_weight <= 0.0:
                    continue
                component = MixtureComponent(
                    dataset_id=dataset_id,
                    split=split,
                    subset=subset,
                    class_field=class_field,
                    # str() unifies the JSON ("0") and in-memory (0) forms.
                    class_value=str(raw_value),
                )
                out.append((component, entry_weight * float(class_weight)))
        else:
            out.append(
                (
                    MixtureComponent(
                        dataset_id=dataset_id, split=split, subset=subset
                    ),
                    float(entry_weight),
                )
            )
    return out


def ground_truth_weights(
    recipes: Mapping[ModelID, Any],
    vertices: Sequence[str] | None = None,
) -> tuple[list[str], dict[ModelID, np.ndarray]]:
    """Put every recipe's mixture on one shared vertex basis.

    The basis is the union of component keys across the whole collection, sorted
    for determinism.  A component a given recipe never draws from gets weight
    0.0 — which is what makes a pure-topic-0 recipe, whose stored
    ``normalized_class_weights`` is simply ``{"0": 1.0}`` with no key for topic 1,
    come back as ``[1.0, 0.0]`` rather than as a one-element vector that cannot
    be compared with its neighbours.

    Parameters
    ----------
    recipes:
        Model ID → recipe, in any of the three accepted forms.
    vertices:
        Optional explicit basis, to compare collections that do not cover the
        same components.  Components outside it are an error rather than a
        silent drop, since dropping one would renormalise every other weight.

    Returns
    -------
    vertices, weights:
        The vertex keys, and one ``(k,)`` array per model in that order.
    """
    if not recipes:
        raise ValueError("no recipes given")

    per_model = {mid: mixture_weights(r) for mid, r in recipes.items()}

    if vertices is None:
        vertex_list = sorted({k for w in per_model.values() for k in w})
    else:
        vertex_list = list(vertices)
        unknown = {
            k for w in per_model.values() for k in w if k not in set(vertex_list)
        }
        if unknown:
            raise ValueError(
                f"components not present in the supplied vertex basis: "
                f"{sorted(unknown)}. Dropping them would renormalise the "
                "remaining weights and silently change the ground truth."
            )

    _reject_split_and_whole(per_model)

    index = {k: i for i, k in enumerate(vertex_list)}
    out: dict[ModelID, np.ndarray] = {}
    for mid, w in per_model.items():
        vec = np.zeros(len(vertex_list), dtype=np.float64)
        for key, value in w.items():
            vec[index[key]] = value
        out[mid] = vec
    return vertex_list, out


def _reject_split_and_whole(per_model: Mapping[ModelID, Mapping[str, float]]) -> None:
    """Refuse a basis where one dataset appears both split by class and whole.

    Taking a dataset as a single vertex is fine, and splitting it into one vertex
    per class is fine, but doing both across one collection is not: the whole-
    dataset vertex is then an *unknown mixture* of the per-class vertices rather
    than independent of them.  Treating all three as corners of a simplex would
    place the models on a geometry that does not exist, and the error would be
    invisible in the output — so it is caught here instead.
    """
    split_bases: dict[str, set[str]] = {}
    whole: set[str] = set()
    for weights in per_model.values():
        for key in weights:
            base, sep, _ = key.partition("[")
            if sep:
                split_bases.setdefault(base, set()).add(key)
            else:
                whole.add(key)

    clash = sorted(whole & set(split_bases))
    if clash:
        example = clash[0]
        raise ValueError(
            f"dataset {example!r} appears both as a whole-dataset component and "
            f"split by class into {sorted(split_bases[example])}. Those are not "
            "independent simplex vertices — the whole-dataset entry is an unknown "
            "mixture of the per-class ones. Give every recipe in the collection "
            "the same treatment of this dataset: set class_filter/class_weights "
            "on all of them, or on none."
        )


def truth_matrix(
    weights: Mapping[ModelID, np.ndarray], model_ids: Sequence[ModelID]
) -> np.ndarray:
    """Stack per-model weight vectors into the ``(n, k)`` array the analysis wants.

    This is the form :func:`src.analysis.simplex.anchor_weight_vs_truth` compares
    against when the mixture has three or more components, and the form
    :func:`simplex_geometry` embeds.
    """
    missing = [m for m in model_ids if m not in weights]
    if missing:
        raise ValueError(f"no ground-truth weights for {missing}")
    return np.vstack([np.asarray(weights[m], dtype=np.float64) for m in model_ids])


# ── the simplex itself ─────────────────────────────────────────────────────────

def simplex_vertices(k: int) -> np.ndarray:
    """Coordinates of a regular ``(k-1)``-simplex with unit edge length.

    Returns a ``(k, k-1)`` array, one row per vertex, all pairwise distances
    equal to 1.

    Built by centering the ``k`` standard basis vectors — which already form a
    regular simplex, just embedded in one dimension more than they need — and
    reading off their coordinates in the rank-``(k-1)`` right-singular basis.
    That drops the redundant dimension without touching any distance, since the
    basis is orthonormal.
    """
    if k < 2:
        raise ValueError(
            f"a simplex needs at least 2 vertices, got {k}. A recipe with one "
            "mixture component has no ground-truth geometry — every model would "
            "sit at the same point."
        )
    centered = np.eye(k) - 1.0 / k
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u[:, : k - 1] * s[: k - 1]
    edge = float(np.linalg.norm(coords[0] - coords[1]))
    return coords / edge


def simplex_geometry(
    weights: Mapping[ModelID, np.ndarray] | np.ndarray,
    model_ids: Sequence[ModelID],
    vertices: Sequence[str],
) -> GeometryResult:
    """Where each model sits on the ground-truth simplex.

    The coordinates are ``W @ V``: each model's mixture weights applied to the
    regular simplex's vertices, which is precisely what "barycentric coordinates"
    means.  A pure recipe lands exactly on a vertex, a 25/75 mixture a quarter of
    the way along the edge.

    The result is a plain :class:`~src.core.geometry.GeometryResult`, so it goes
    straight into :func:`~src.analysis.configurations.procrustes_compare` and
    :func:`~src.analysis.configurations.protest` against any taxonomy's
    embedding.  Unlike the barycentric route this needs no anchor models, so it
    still works on a collection with no pure endpoints.
    """
    W = weights if isinstance(weights, np.ndarray) else truth_matrix(weights, model_ids)
    W = np.asarray(W, dtype=np.float64)
    k = len(vertices)
    if W.shape != (len(model_ids), k):
        raise ValueError(
            f"weights shape {W.shape} expected ({len(model_ids)}, {k})"
        )

    coords = W @ simplex_vertices(k)
    return GeometryResult(
        coordinates=coords.astype(np.float32),
        model_ids=list(model_ids),
        method="simplex",
        taxonomy="ground_truth",
        n_components=k - 1,
        stress=0.0,  # exact by construction — nothing was fitted
        metadata={"vertices": list(vertices)},
    )


def simplex_distance_matrix(
    weights: Mapping[ModelID, np.ndarray] | np.ndarray,
    model_ids: Sequence[ModelID],
    vertices: Sequence[str],
) -> DistanceMatrix:
    """Pairwise Euclidean distances on the ground-truth simplex.

    The matrix-level counterpart to :func:`simplex_geometry`, for
    :func:`~src.analysis.matrices.matrix_correlation` and
    :func:`~src.analysis.matrices.mantel_test` against a taxonomy's distances.
    """
    from scipy.spatial.distance import pdist, squareform

    geo = simplex_geometry(weights, model_ids, vertices)
    matrix = squareform(pdist(np.asarray(geo.coordinates, dtype=np.float64)))
    return DistanceMatrix(
        matrix=matrix,
        model_ids=list(model_ids),
        metric="euclidean",
        taxonomy="ground_truth",
    )


# ── scoring a taxonomy against the truth ──────────────────────────────────────

def dcor_vs_truth(dm: DistanceMatrix, truth_dm: DistanceMatrix) -> float:
    """Bias-corrected distance correlation against the ground-truth simplex.

    :func:`~src.analysis.matrices.distance_correlation` returns a bare float; it
    is ``dcor_test`` that returns a result object carrying ``.statistic``. Note
    the U-centred dCor* lives on a squared scale and may legitimately be
    negative, so do not clip it.

    A *matrix-level* score: it reads the distances and never embeds, so unlike
    :func:`disparity_vs_truth` it is untouched by MDS distortion. It is also
    invariant to a constant rescaling of either matrix, which is why
    :func:`simplex_distance_matrix` and a plain ``pdist`` over the raw weight
    vectors — which differ by exactly a factor of ``1/√2`` — score identically.
    """
    from .matrices import distance_correlation

    return float(distance_correlation(dm, truth_dm))


def disparity_vs_truth(
    dm: DistanceMatrix,
    truth_geometry: GeometryResult,
    *,
    geometry: GeometryResult | None = None,
    random_state: int = 0,
    n_components: int = 2,
) -> float:
    """Scaled residual Procrustes disparity of *dm*'s embedding against the truth.

    Embeds *dm* with MDS and superimposes it on *truth_geometry*, returning
    :attr:`~src.analysis.configurations.ProcrustesResult.disparity` — the
    residual sum of squares as a fraction of total squared coordinate variance,
    in ``[0, 1]`` with **0 meaning identical shape**. It therefore runs opposite
    to :func:`dcor_vs_truth`, which is worth saying out loud wherever the two are
    reported side by side.

    Where ``dcor_vs_truth`` scores the *distances*, this scores the
    *configuration* — the arrangement of points an embedding actually draws.
    The two come apart: a taxonomy can reproduce the pairwise distance profile
    while arranging the points in something that is not the simplex.

    The price of that is a dependency on the embedding. This number is
    MDS-mediated where ``dcor_vs_truth`` is not, so it inherits whatever
    distortion the projection introduced — read it beside
    :func:`~src.analysis.quality.kruskal_stress`, not instead of it. For the same
    reason *random_state* is an explicit argument rather than a hidden default:
    :class:`~src.geometry_methods.mds.MDSGeometry` initialises randomly, so a
    caller wanting this score to describe the configuration it is *plotting*
    must pass the seed that configuration was fitted under.

    The ground truth goes in as the first argument to
    :func:`~src.analysis.configurations.procrustes_compare`, matching
    ``compare_taxonomies``. Disparity is symmetric under the swap, so no number
    changes; the orientation only fixes the fitted map to run taxonomy →
    ground-truth frame, which is the useful direction.

    Models are paired by identifier, not by row position: ``procrustes_compare``
    reindexes both configurations onto their common ``model_ids`` first. So
    permuting *dm*'s rows together with its ids leaves this unchanged, and
    mislabelled rows are a genuine disagreement rather than a silent one.

    Pass *geometry* to score an embedding you have already fitted — the one you
    are plotting, or one you also want the stress of — instead of fitting a
    second one here. *dm* is then unused, and *random_state* / *n_components*
    describe nothing, so it is on the caller to pass the geometry that actually
    came from *dm*.
    """
    from .bridge import fit_geometry
    from .configurations import procrustes_compare

    geo = geometry
    if geo is None:
        geo = fit_geometry(dm, method="mds", n_components=n_components,
                           random_state=random_state)
    return float(procrustes_compare(truth_geometry, geo).disparity)


# ── which models play which role ───────────────────────────────────────────────

def pure_anchors(
    vertices: Sequence[str],
    weights: Mapping[ModelID, np.ndarray],
    prefer: Sequence[ModelID] | None = None,
    tol: float = 1e-9,
) -> list[ModelID]:
    """One model per vertex, being the one trained on that pure component.

    These are the anchors :func:`~src.analysis.simplex.barycentric` needs: models
    whose barycentric coordinates are known to be one-hot, so they pin the
    measured simplex to the true one.  Returned in ``vertices`` order, so the
    weight columns line up with the ground-truth columns with no reordering.

    Parameters
    ----------
    prefer:
        Tie-break order, used when several models share a vertex.  That is the
        normal situation in a pooled collection — four adapters trained on 100%
        topic 0 at different sample sizes all sit at the same vertex of the
        *truth* while occupying four different points in the *geometry*.  Pass an
        ordering (largest sample size first, say) to choose deterministically.
        Without it, ambiguity is an error rather than an arbitrary pick.
    """
    order = {m: i for i, m in enumerate(prefer)} if prefer else {}
    anchors: list[ModelID] = []

    for j, vertex in enumerate(vertices):
        candidates = [
            m
            for m, w in weights.items()
            if abs(float(w[j]) - 1.0) <= tol and float(np.abs(w).sum() - w[j]) <= tol
        ]
        if not candidates:
            raise ValueError(
                f"no model is trained purely on component {vertex!r}, so it "
                "cannot anchor the simplex. Either add that pure model to the "
                "collection, pass anchors explicitly, or use the Procrustes "
                "route (simplex_geometry), which needs no anchors."
            )
        if len(candidates) > 1 and not order:
            raise ValueError(
                f"{len(candidates)} models are pure in component {vertex!r}: "
                f"{sorted(candidates)}. Pass prefer=... to break the tie "
                "deterministically, or anchors=... to choose explicitly."
            )
        anchors.append(
            min(candidates, key=lambda m: (order.get(m, len(order)), str(m)))
        )

    if len(set(anchors)) != len(anchors):
        raise ValueError(
            f"the same model anchors more than one vertex: {anchors}. Its "
            "ground-truth weights are not one-hot in a single component."
        )
    return anchors


def evaluation_points(
    model_ids: Iterable[ModelID], anchors: Sequence[ModelID]
) -> list[ModelID]:
    """The non-anchor models — where the geometry is actually being tested.

    Anchors are one-hot *by construction*, so a correlation computed over them
    partly measures the projection's own definition.  The mixtures lying along
    the simplex edges are the points that carry information, and scoring them
    separately is what keeps the recovery number honest.
    """
    anchor_set = set(anchors)
    return [m for m in model_ids if m not in anchor_set]
