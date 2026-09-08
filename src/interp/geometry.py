"""Do the fitted coefficients reproduce the *shape* of the ground-truth simplex?

:mod:`src.interp.fitting` scores each adapter on its own: ``coeff_l2 = ‖ŵ − w‖₂``
is a per-point error, one number per mixture.  This module asks the
configuration-level question instead — taken together as a point cloud, do the
fitted coefficients lay out the same geometry as the data recipes do?

Those are genuinely different questions.  A set of fits can each be individually
off yet still reproduce the arrangement exactly (a uniform rotation of the whole
cloud), or each be individually close while scrambling which mixture sits next
to which.  Only the second question is comparable to what
:mod:`src.analysis.comparison` reports for the structural, behavioral and
functional levels, so answering it here is what puts coefficient recovery on the
same scale as the rest of the taxonomy work.

Two measurements, each against the ideal simplex, for each of the two
coefficient estimates (``ols_normalized`` and ``simplex``):

- **Procrustes disparity** — configurations, after optimal
  translation/rotation/scale.  0 is identical shape.  The discriminating one.
- **dCor** — distance matrices directly, no embedding.  Corroboration only: at
  the sizes here it saturates near 1 and separates the two estimates by ~0.006.
  It is also *unsigned*, so it cannot see a reversed geometry; see
  :func:`src.analysis.matrices.distance_correlation`.

Nothing here is new math.  Every statistic comes from :mod:`src.analysis`; this
module's job is to feed the coefficient matrices into it on exactly the same
footing as the ground truth, and to record what the numbers do and do not mean.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "COEFF_KINDS",
    "CoefficientGeometry",
    "CoefficientGeometryReport",
    "coefficient_distance_matrix",
    "compare_coefficients",
    "compare_from_fits",
    "compare_from_report",
]

SCHEMA_VERSION = "1"

#: The two coefficient estimates worth scoring, both of which sum to 1.  The raw
#: unconstrained ``ols`` is deliberately absent — it sums to γ ≈ 0.73, which the
#: barycentric embedding projects away, so scoring it here would silently
#: measure something other than what it says.  γ is reported as its own scalar
#: by :class:`~src.interp.fitting.InterpolationFit`.
COEFF_KINDS = ("ols_normalized", "simplex")

TAXONOMY = "coefficients"


def coefficient_distance_matrix(
    weights: np.ndarray,
    model_ids: Sequence[str],
    vertices: Sequence[str],
    taxonomy: str = TAXONOMY,
):
    """Pairwise distances between coefficient vectors, in ground-truth units.

    Delegates to :func:`src.analysis.ground_truth.simplex_distance_matrix` —
    the same function that builds the truth side of every comparison here — and
    only retags the result, so a saved matrix is not mislabelled
    ``ground_truth``.

    That function reads each row as *barycentric coordinates*: a vector like
    ``[0.25, 0.50, 0.25]`` is three numbers under one constraint, so it has only
    ``k-1`` real degrees of freedom, and ``W @ simplex_vertices(k)`` places it at
    the corresponding point of a unit-edge regular simplex.  It is the same
    operation :func:`src.interp.plots._project` performs for the ternary plot.

    **This does not change any reported number.**  For rows summing to 1 the
    embedding is an isometry up to a constant — ``‖(w₁−w₂) @ V‖ = ‖w₁−w₂‖/√2``,
    since ``simplex_vertices`` divides by the unit edge ``√2`` — and both
    Procrustes (with ``scaling=True``) and dCor are scale-invariant, so the
    constant washes out.  Measured on the real collection, going through the
    embedding and taking ``pdist(W)`` directly agree to 8 decimals on both
    statistics.  The embedding is used anyway for two reasons: both sides of the
    comparison then come from one function, with no way for the fitted distances
    to end up a factor of √2 from the truth distances in a saved artifact; and
    it drops the redundant dimension, so the MDS at ``n_components = k-1`` is
    embedding something that genuinely is ``(k-1)``-dimensional.

    The isometry holds **only because the rows sum to 1**.  For weights that do
    not — the raw ``ols``, summing to γ — the embedding projects out the γ
    direction and distances shift materially.  Pinned by
    ``t_coefficient_dm_isometry``.
    """
    from src.analysis.ground_truth import simplex_distance_matrix

    dm = simplex_distance_matrix(weights, list(model_ids), list(vertices))
    return replace(dm, taxonomy=taxonomy)


@dataclass
class CoefficientGeometry:
    """The measurements for one coefficient estimate against the truth."""

    kind: str
    distance_matrix: object            # DistanceMatrix over the coefficients
    geometry: object                   # GeometryResult: the MDS embedding
    procrustes: object                 # ProcrustesResult vs the truth simplex
    dcor: object                       # DcorResult vs the truth distances
    #: Disparity computed straight from the barycentric embedding, skipping the
    #: MDS.  The check that the embedding step did not distort the shape: at
    #: ``n_components = k-1`` the distance matrix is exactly realisable, so this
    #: should match :attr:`procrustes` to a few decimals.
    procrustes_direct: float
    #: The embedding's stress, or None for a method that does not fit one
    #: (classical MDS solves in closed form).  Not coerced to 0.0: that would
    #: render as a perfect embedding rather than as "not measured".
    stress: float | None
    #: ``(n,)`` per-model distance between the two superimposed positions, in
    #: :attr:`model_ids` order.  Which adapters disagree, not just how much.
    residuals: np.ndarray
    protest: object | None = None      # ProtestResult, when run

    @property
    def model_ids(self) -> list[str]:
        return list(self.procrustes.model_ids)

    def summary(self) -> dict:
        return {
            "kind": self.kind,
            "procrustes_disparity": float(self.procrustes.disparity),
            "procrustes_direct": float(self.procrustes_direct),
            "protest_p": None if self.protest is None else float(self.protest.p_value),
            "dcor": float(self.dcor.statistic),
            "dcor_p": float(self.dcor.p_value),
            "dcor_bias_corrected": bool(self.dcor.bias_corrected),
            "mds_stress": None if self.stress is None else float(self.stress),
            "residual_max": float(self.residuals.max()),
            "residual_mean": float(self.residuals.mean()),
        }


@dataclass
class CoefficientGeometryReport:
    """Every coefficient estimate scored against one ground-truth simplex."""

    vertices: list[str]
    model_ids: list[str]
    labels: dict[str, str]
    is_corner: list[bool]
    truth_geometry: object             # GeometryResult
    truth_matrix: object               # DistanceMatrix
    results: dict[str, CoefficientGeometry]
    config: dict = field(default_factory=dict)

    def headline(self) -> dict:
        """The four numbers this module exists to produce."""
        return {
            kind: {
                "procrustes_disparity": float(r.procrustes.disparity),
                "dcor": float(r.dcor.statistic),
            }
            for kind, r in self.results.items()
        }

    # ── rendering ────────────────────────────────────────────────────────────

    def to_report(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "vertices": self.vertices,
            "model_ids": self.model_ids,
            "labels": [self.labels[m] for m in self.model_ids],
            "is_corner": self.is_corner,
            "n_models": len(self.model_ids),
            "n_corners": int(sum(self.is_corner)),
            "config": self.config,
            "headline": self.headline(),
            "results": {k: r.summary() for k, r in self.results.items()},
            "per_model_residuals": {
                k: r.residuals.tolist() for k, r in self.results.items()
            },
        }

    def to_markdown(self, figures: bool = True) -> str:
        n, n_corner = len(self.model_ids), int(sum(self.is_corner))
        cfg = self.config
        label = self.labels

        lines = [
            "# Coefficient geometry vs the ground-truth simplex",
            "",
            "## What this measures",
            "",
            "The interpolation fit gives every adapter a set of coefficients — how "
            "much of each pure-component adapter it looks like. The data recipe "
            "gives every adapter its *true* proportions. Both are points on the "
            "same triangle.",
            "",
            "`report.md` already asks how far each fitted point is from its own "
            "true point, one adapter at a time. This asks a different question: "
            "**do the fitted points, all together, form the same arrangement as "
            "the true ones?** A set of fits could each be a little off and still "
            "lay out the same shape — or each be close while scrambling which "
            "adapter neighbours which. Only the second question is comparable to "
            "what the other taxonomy levels report.",
            "",
            "Two statistics, both against the ideal simplex the recipes define:",
            "",
            "- **Procrustes disparity** — slide, rotate and rescale the fitted "
            "arrangement to sit on top of the true one as well as it can, then "
            "measure what is left over. **0 = identical shape**, 1 = no better "
            "than unrelated. This is the number that discriminates.",
            "- **dCor** — distance correlation, computed on the distance matrices "
            "with no embedding step at all. **1 = perfect dependence.** Read it "
            "as corroboration, not as the headline: see the caveats below.",
            "",
            "## The four numbers",
            "",
            "| coefficients | Procrustes disparity ↓ | dCor ↑ |",
            "|---|---|---|",
        ]
        for kind in COEFF_KINDS:
            r = self.results.get(kind)
            if r is None:
                continue
            lines.append(
                f"| `{kind}` | {r.procrustes.disparity:.5f} | {r.dcor.statistic:.4f} |"
            )

        lines += [
            "",
            f"Measured over **{n} adapters**"
            + (
                f", including the {n_corner} corner adapters."
                if n_corner
                else " — corner adapters excluded."
            ),
        ]
        if n_corner:
            lines += [
                "",
                f"> Worth knowing when reading these: a corner adapter fitted "
                f"against the corner basis returns itself exactly, so {n_corner} of "
                f"the {n} points are perfect by construction and sit on the "
                f"simplex's vertices. They pin the arrangement and flatter both "
                f"scores. Re-run with `--mixtures-only` for the "
                f"{n - n_corner}-mixture configuration, where every point is a "
                f"genuine fit.",
            ]

        lines += [
            "",
            "## Full results",
            "",
            "| coefficients | disparity | disparity (no MDS) | PROTEST p | dCor | "
            "dCor p | MDS stress | worst model |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for kind in COEFF_KINDS:
            r = self.results.get(kind)
            if r is None:
                continue
            worst = r.model_ids[int(np.argmax(r.residuals))]
            p = "—" if r.protest is None else f"{r.protest.p_value:.4g}"
            lines.append(
                f"| `{kind}` | {r.procrustes.disparity:.5f} "
                f"| {r.procrustes_direct:.5f} | {p} "
                f"| {r.dcor.statistic:.4f} | {r.dcor.p_value:.4g} "
                f"| {'—' if r.stress is None else format(r.stress, '.2e')} "
                f"| `{label.get(worst, worst)}` "
                f"({r.residuals.max():.4f}) |"
            )

        if figures:
            lines += [
                "",
                "## Figure",
                "",
                "![Procrustes superposition](figures/procrustes_superposition.png)",
                "",
                "**`procrustes_superposition.png`** — the two arrangements laid on "
                "top of each other after the best possible alignment. Hollow "
                "markers are the true recipes, filled markers the fitted "
                "coefficients, and each connector is one adapter's leftover error. "
                "Short connectors everywhere means the shape was recovered; one "
                "long connector in an otherwise tidy panel means a single adapter "
                "is responsible for the disparity, which the summary number alone "
                "would hide.",
            ]

        lines += [
            "",
            "## Reading this",
            "",
            self._interpretation(),
            "",
            "## Caveats",
            "",
            self._embedding_caveat(),
            "- **dCor is unsigned and saturates.** It measures dependence, not "
            "agreement: an arrangement that reproduces the truth exactly backwards "
            "scores 1.0, identically to a perfect one. At these sizes it also sits "
            "near the top of its range for both estimates, separating them by less "
            "than 0.01. Rank by Procrustes; treat dCor as a sanity check that the "
            "structure is present at all.",
            "- **The dCor p-value floors at "
            f"1/(n_permutations+1) = {1 / (cfg.get('n_permutations', 9999) + 1):.1e}**, "
            "so `p` at that value means *no permutation did better*, not a "
            "precisely estimated tail.",
            "",
            "## Provenance",
            "",
            f"- Coefficient distances: barycentric embedding of the fitted weights, "
            f"the same construction used for the truth "
            f"(`src.analysis.ground_truth.simplex_distance_matrix`)",
            f"- Embedding: `{cfg.get('method')}`, "
            f"n_components={cfg.get('n_components')}, "
            f"random_state={cfg.get('random_state')}",
            f"- Permutations: {cfg.get('n_permutations')}",
            f"- Simplex vertices: {', '.join(self.vertices)}",
            "",
        ]
        return "\n".join(lines) + "\n"

    def _embedding_caveat(self) -> str:
        """State what the embedding step actually did, from the measured values.

        Written from the results rather than hardcoded, because the honest
        answer changes with the configuration: at ``n_components = k-1`` the
        distances are exactly realisable and the step is a formality, while
        below that it is genuinely projecting and the disparity it reports is
        partly a property of the projection.
        """
        cfg = self.config
        exact = cfg.get("n_components") == len(self.vertices) - 1
        method = cfg.get("method", "mds")
        name = "classical MDS" if method == "pca" else "MDS"

        gaps = [
            abs(r.procrustes.disparity - r.procrustes_direct)
            for r in self.results.values()
        ]
        stresses = [r.stress for r in self.results.values() if r.stress is not None]
        evidence = (
            f"the disparity it reports differs from the one computed straight "
            f"from the coefficients by at most {max(gaps):.1e}"
            if gaps
            else "no estimates were scored"
        )
        if stresses:
            evidence += f", at a stress of {max(stresses):.1e}"

        if exact:
            return (
                f"- **The {name} step is close to a no-op here.** With "
                f"{len(self.vertices)} components the coefficient distance matrix "
                f"is exactly Euclidean in {len(self.vertices) - 1} dimensions, so "
                f"embedding it at `n_components={cfg.get('n_components')}` "
                f"reproduces the arrangement rather than approximating it — "
                f"{evidence}. The step is kept because it is what makes these "
                f"numbers commensurate with the other taxonomy levels, and it "
                f"begins doing real work at more components or a lower projection "
                f"dimension. It is not doing any here, and this report should not "
                f"imply otherwise."
            )
        return (
            f"- **The {name} step is doing real work here.** "
            f"`n_components={cfg.get('n_components')}` is below the "
            f"{len(self.vertices) - 1} dimensions the coefficient distances "
            f"actually span, so the embedding is projecting rather than "
            f"reproducing, and {evidence}. Part of the disparity below is that "
            f"projection rather than coefficient error; the *disparity (no MDS)* "
            f"column is the projection-free comparison."
        )

    def _interpretation(self) -> str:
        ranked = sorted(
            self.results.items(), key=lambda kv: kv[1].procrustes.disparity
        )
        if not ranked:
            return "No coefficient estimates were scored."

        best_kind, best = ranked[0]
        parts = [
            f"**The fitted coefficients recover the simplex.** `{best_kind}` reaches "
            f"a disparity of {best.procrustes.disparity:.5f} — on a scale where 0 is "
            f"an identical arrangement and 1 is no relationship — with "
            f"dCor {best.dcor.statistic:.4f}. The geometry the data mixtures define "
            f"is present in the adapter weights, not just approximately but as very "
            f"nearly the same shape."
        ]

        if len(ranked) > 1:
            worst_kind, worst = ranked[-1]
            ratio = (
                worst.procrustes.disparity / best.procrustes.disparity
                if best.procrustes.disparity > 0
                else float("inf")
            )
            parts.append(
                f"**Rescaling beats constraining, here too.** `{worst_kind}` scores "
                f"{worst.procrustes.disparity:.5f}, {ratio:.1f}x worse than "
                f"`{best_kind}`. That agrees with the per-adapter finding in "
                f"`report.md` — forcing the coefficients to sum to 1 fights the "
                f"shrinkage γ instead of reporting it, and pushes the surplus onto "
                f"components that should be zero. The same defect that roughly "
                f"doubles the per-point error also deforms the arrangement, so the "
                f"two measurements are telling one story rather than two."
            )

        n_corner = int(sum(self.is_corner))
        if n_corner:
            parts.append(
                f"**Read the absolute value with the corners in mind.** "
                f"{n_corner} of {len(self.model_ids)} points are corner adapters, "
                f"which fit themselves exactly and sit on the simplex's vertices. "
                f"They anchor the arrangement, so the disparity here is a floor "
                f"rather than a neutral estimate. The ratio between the two "
                f"estimates is the robust part; the mixtures-only run is the "
                f"stricter test of the absolute number."
            )

        return "\n\n".join(parts)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path, figures: bool = True) -> Path:
        """Write ``coefficient_geometry.{json,md}`` plus the arrays.

        Mirrors :meth:`src.interp.report.InterpolationReport.save`: JSON for
        scalars, safetensors at float64 for arrays, markdown for humans.

        Stored directly rather than through ``DistanceMatrix.save`` /
        ``GeometryResult.save`` because those cast to float32 on the way out.
        That would be a *second* truncation: ``simplex_geometry`` already stores
        its coordinates as float32, so these arrays carry float32 precision
        (~1e-7 on O(1) distances) however they are written.  Writing float64
        preserves exactly the values that were computed instead of rounding them
        again, which is what makes a byte-for-byte round-trip check meaningful.
        The precision floor is far below the disparities being reported and
        affects no result.
        """
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "coefficient_geometry.json").write_text(
            json.dumps(self.to_report(), indent=2)
        )
        (path / "coefficient_geometry.md").write_text(self.to_markdown(figures=figures))

        arrays = {
            "truth_coordinates": np.ascontiguousarray(
                self.truth_geometry.coordinates, dtype=np.float64
            ),
            "truth_distances": np.ascontiguousarray(
                self.truth_matrix.matrix, dtype=np.float64
            ),
        }
        for kind, r in self.results.items():
            arrays[f"{kind}/distances"] = np.ascontiguousarray(
                r.distance_matrix.matrix, dtype=np.float64
            )
            arrays[f"{kind}/coordinates"] = np.ascontiguousarray(
                r.geometry.coordinates, dtype=np.float64
            )
            arrays[f"{kind}/aligned_truth"] = np.ascontiguousarray(
                r.procrustes.aligned_a.coordinates, dtype=np.float64
            )
            arrays[f"{kind}/aligned_fit"] = np.ascontiguousarray(
                r.procrustes.aligned_b.coordinates, dtype=np.float64
            )
            arrays[f"{kind}/residuals"] = np.ascontiguousarray(
                r.residuals, dtype=np.float64
            )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "model_ids": self.model_ids,
            "labels": [self.labels[m] for m in self.model_ids],
            "is_corner": self.is_corner,
            "vertices": self.vertices,
            "kinds": list(self.results),
            "config": self.config,
        }
        arrays["_meta_json"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8
        )
        geo_dir = path / "coefficient_geometry"
        geo_dir.mkdir(parents=True, exist_ok=True)
        save_file(arrays, str(geo_dir / "geometry.safetensors"))
        return path


def compare_coefficients(
    coeffs: Mapping[str, np.ndarray],
    true: np.ndarray,
    model_ids: Sequence[str],
    vertices: Sequence[str],
    *,
    labels: Mapping[str, str] | None = None,
    is_corner: Sequence[bool] | None = None,
    kinds: Sequence[str] = COEFF_KINDS,
    n_components: int | None = None,
    method: str = "mds",
    n_permutations: int = 9999,
    random_state: int = 0,
    run_protest: bool = True,
) -> CoefficientGeometryReport:
    """Score each coefficient estimate against the ground-truth simplex.

    *coeffs* maps a kind name (``"ols_normalized"``, ``"simplex"``) to its
    ``(n, k)`` matrix; *true* is the ``(n, k)`` matrix of data proportions.  All
    are row-aligned to *model_ids*, and *vertices* names the ``k`` components.

    *n_components* defaults to ``k-1``, the dimension in which the coefficient
    distances are exactly realisable.  *method* is anything
    :func:`src.analysis.bridge.fit_geometry` accepts; ``"pca"`` is classical MDS,
    deterministic and dependency-free, and is exact for these matrices.
    """
    from src.analysis.bridge import fit_geometry
    from src.analysis.configurations import (
        per_point_residuals,
        procrustes_compare,
        protest,
    )
    from src.analysis.ground_truth import simplex_geometry
    from src.analysis.matrices import dcor_test

    model_ids = list(model_ids)
    vertices = list(vertices)
    n, k = len(model_ids), len(vertices)
    true = np.asarray(true, dtype=np.float64)
    if true.shape != (n, k):
        raise ValueError(f"true has shape {true.shape}, expected ({n}, {k})")
    if n < 4:
        # dcor_test's own floor for the bias-corrected statistic; failing here
        # names the cause rather than surfacing it three frames down.
        raise ValueError(
            f"need at least 4 models for the bias-corrected dCor, got {n}. "
            "A configuration this small has no shape to compare."
        )

    n_components = k - 1 if n_components is None else n_components
    if not 1 <= n_components <= k - 1:
        raise ValueError(
            f"n_components={n_components} outside [1, {k - 1}]; the simplex for "
            f"{k} components spans exactly {k - 1} dimensions."
        )

    truth_geometry = simplex_geometry(true, model_ids, vertices)
    truth_matrix = coefficient_distance_matrix(
        true, model_ids, vertices, taxonomy="ground_truth"
    )

    geo_kwargs = {} if method == "pca" else {"random_state": random_state}

    results: dict[str, CoefficientGeometry] = {}
    for kind in kinds:
        if kind not in coeffs:
            raise KeyError(f"no coefficients supplied for {kind!r}")
        W = np.asarray(coeffs[kind], dtype=np.float64)
        if W.shape != (n, k):
            raise ValueError(f"{kind} has shape {W.shape}, expected ({n}, {k})")
        if not np.isfinite(W).all():
            raise ValueError(
                f"{kind} contains non-finite values; a fit with gamma ~ 0 leaves "
                "NaNs in ols_normalized and cannot be placed on the simplex."
            )

        dm = coefficient_distance_matrix(W, model_ids, vertices)
        geo = fit_geometry(dm, method=method, n_components=n_components, **geo_kwargs)

        pr = procrustes_compare(truth_geometry, geo)
        # The same comparison with the embedding skipped: at n_components = k-1
        # these must agree, and a gap means the embedding lost something.
        pr_direct = procrustes_compare(
            truth_geometry, simplex_geometry(W, model_ids, vertices)
        )
        pt = (
            protest(
                truth_geometry,
                geo,
                n_permutations=n_permutations,
                random_state=random_state,
            )
            if run_protest
            else None
        )
        dc = dcor_test(
            dm,
            truth_matrix,
            n_permutations=n_permutations,
            random_state=random_state,
        )

        results[kind] = CoefficientGeometry(
            kind=kind,
            distance_matrix=dm,
            geometry=geo,
            procrustes=pr,
            protest=pt,
            dcor=dc,
            procrustes_direct=float(pr_direct.disparity),
            stress=None if geo.stress is None else float(geo.stress),
            residuals=per_point_residuals(pr),
        )

    return CoefficientGeometryReport(
        vertices=vertices,
        model_ids=model_ids,
        labels=dict(labels) if labels else {m: m for m in model_ids},
        is_corner=[bool(c) for c in (is_corner or [False] * n)],
        truth_geometry=truth_geometry,
        truth_matrix=truth_matrix,
        results=results,
        config={
            "n_components": n_components,
            "method": method,
            "random_state": random_state,
            "n_permutations": n_permutations,
            "n_models": n,
            "n_corners": int(sum(is_corner)) if is_corner else 0,
            "kinds": list(kinds),
        },
    )


def compare_from_fits(
    fits: Mapping, mixtures_only: bool = False, **kwargs
) -> CoefficientGeometryReport:
    """Score the arrays :func:`src.interp.report.load_fits` returns.

    The entry point that needs no adapters and no Gram: everything it reads is
    already in ``fits/fits.safetensors`` from the interpolation run.

    With *mixtures_only* the corner adapters are dropped.  They fit themselves
    exactly and land on the simplex's vertices, so they anchor the arrangement
    and flatter both statistics; excluding them scores only genuine mixtures.
    """
    keep = [
        i
        for i, corner in enumerate(fits["is_corner"])
        if not (mixtures_only and corner)
    ]
    if mixtures_only and len(keep) == len(fits["is_corner"]):
        raise ValueError(
            "mixtures_only was requested but no fit is flagged as a corner; "
            "the run has no corner adapters to drop."
        )
    idx = np.asarray(keep, dtype=int)
    model_ids = [fits["targets"][i] for i in keep]

    return compare_coefficients(
        {kind: np.asarray(fits[kind])[idx] for kind in COEFF_KINDS if kind in fits},
        np.asarray(fits["true"])[idx],
        model_ids,
        fits["vertices"],
        labels=dict(zip(fits["targets"], fits["labels"])),
        is_corner=[bool(fits["is_corner"][i]) for i in keep],
        **kwargs,
    )


def compare_from_report(
    report, mixtures_only: bool = False, **kwargs
) -> CoefficientGeometryReport:
    """Score an in-memory :class:`~src.interp.report.InterpolationReport`.

    The same comparison as :func:`compare_from_fits`, for the run that just
    produced the fits rather than one read back off disk.
    """
    selected = report.mixtures if mixtures_only else report.fits
    if not selected:
        raise ValueError("no fits to score")
    labels = report.labels()

    return compare_coefficients(
        {
            kind: np.stack([getattr(f, kind) for f in selected])
            for kind in COEFF_KINDS
        },
        np.stack([f.true for f in selected]),
        [f.target for f in selected],
        report.vertices,
        labels=labels,
        is_corner=[f.is_corner for f in selected],
        **kwargs,
    )
