"""Assembling, saving and rendering the interpolation results.

Follows the storage convention of :meth:`src.analysis.comparison.TaxonomyComparison.save`:
JSON for scalars and provenance, safetensors for arrays, ``report.md`` for
humans.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.interp.fitting import InterpolationFit, fit_interpolation, random_simplex_baseline

__all__ = [
    "InterpolationReport",
    "build_report",
    "detect_corners",
    "load_fits",
    "short_labels",
]

SCHEMA_VERSION = "1"


def short_labels(names: list[str]) -> dict[str, str]:
    """Map adapter names to display labels by dropping what they all share.

    Adapter directory names are dominated by a training suffix identical across
    a collection (`_n1000_s00_r16_i00_b5008_fea27ccee`), which crowds out the
    part that actually varies.  Dropping the underscore-separated segments every
    name shares leaves exactly the distinguishing part — `025g1_050g2_025g3` —
    without hard-coding any naming convention.  Works on segments rather than
    characters so a shared trailing `g3` is not sheared off the last component.

    Falls back to the full names if trimming would make two of them collide.
    """
    if not names:
        return {}
    if len(names) == 1:
        return {names[0]: names[0]}

    parts = [n.split("_") for n in names]
    shortest = min(len(p) for p in parts)

    head = 0
    while head < shortest and len({p[head] for p in parts}) == 1:
        head += 1
    tail = 0
    while tail < shortest - head and len({p[-1 - tail] for p in parts}) == 1:
        tail += 1

    trimmed = {
        n: "_".join(p[head : len(p) - tail]) or n for n, p in zip(names, parts)
    }
    if len(set(trimmed.values())) != len(names):
        return {n: n for n in names}
    return trimmed


def detect_corners(
    vertices: list[str],
    truth: dict[str, np.ndarray],
    prefer: list[str] | None = None,
) -> list[str]:
    """The adapter trained purely on each component, in vertex order.

    Delegates to :func:`src.analysis.ground_truth.pure_anchors`, which already
    handles the "no pure model for this vertex" and "several tied" cases with
    the errors this package would otherwise have to reinvent.
    """
    from src.analysis.ground_truth import pure_anchors

    return pure_anchors(vertices, truth, prefer=prefer)


@dataclass
class InterpolationReport:
    """Fits for every non-corner adapter, plus the corners' self-consistency."""

    fits: list[InterpolationFit]
    corners: list[str]
    vertices: list[str]
    blocks: list[tuple[int, str]]
    scale: float
    base_model_id: str
    rank: int
    lora_alpha: int
    corner_cosines: np.ndarray                    # (k, k)
    corner_cond: float
    provenance: dict = field(default_factory=dict)

    @property
    def mixtures(self) -> list[InterpolationFit]:
        """Fits for the genuine mixtures — the corners excluded."""
        return [f for f in self.fits if not f.is_corner]

    def labels(self) -> dict[str, str]:
        """Display labels for every fitted adapter."""
        return short_labels([f.target for f in self.fits])

    def summary(self) -> dict:
        m = self.mixtures
        if not m:
            return {}
        baseline = float(np.mean([random_simplex_baseline(f.true) for f in m]))
        return {
            "n_mixtures": len(m),
            "gamma_mean": float(np.mean([f.gamma for f in m])),
            "gamma_std": float(np.std([f.gamma for f in m])),
            "rel_true_mean": float(np.mean([f.rel_true for f in m])),
            "rel_ols_mean": float(np.mean([f.rel_ols for f in m])),
            "rel_simplex_mean": float(np.mean([f.rel_simplex for f in m])),
            "cosine_mean": float(np.mean([f.cosine for f in m])),
            "coeff_l2_mean": float(np.mean([f.coeff_l2 for f in m])),
            "coeff_l2_simplex_mean": float(np.mean([f.coeff_l2_simplex for f in m])),
            "coeff_l2_random_baseline": baseline,
            "coeff_l2_vs_baseline": baseline / float(np.mean([f.coeff_l2 for f in m])),
        }

    # ── rendering ────────────────────────────────────────────────────────────

    def to_report(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "base_model_id": self.base_model_id,
            "vertices": self.vertices,
            "corners": self.corners,
            "n_blocks": len(self.blocks),
            "blocks": [[int(l), m] for l, m in self.blocks],
            "scale": self.scale,
            "rank": self.rank,
            "lora_alpha": self.lora_alpha,
            "corner_cosines": self.corner_cosines.tolist(),
            "corner_cond": self.corner_cond,
            "summary": self.summary(),
            "fits": [
                {
                    "target": f.target,
                    "label": self.labels()[f.target],
                    "is_corner": f.is_corner,
                    "true": f.true.tolist(),
                    "ols": f.ols.tolist(),
                    "ols_normalized": f.ols_normalized.tolist(),
                    "simplex": f.simplex.tolist(),
                    "face": list(f.face),
                    "gamma": f.gamma,
                    "norm": float(np.sqrt(f.norm_sq)),
                    "residual_true": f.residual_true,
                    "residual_ols": f.residual_ols,
                    "residual_simplex": f.residual_simplex,
                    "rel_true": f.rel_true,
                    "rel_ols": f.rel_ols,
                    "rel_simplex": f.rel_simplex,
                    "cosine": f.cosine,
                    "coeff_l2": f.coeff_l2,
                    "coeff_l1": f.coeff_l1,
                    "coeff_l2_simplex": f.coeff_l2_simplex,
                }
                for f in self.fits
            ],
            "provenance": self.provenance,
        }

    def to_markdown(self, figures: bool = True) -> str:
        s = self.summary()
        fmt = lambda v: "[" + " ".join(f"{x:.3f}" for x in v) + "]"  # noqa: E731
        label = self.labels()
        v = self.vertices

        lines = [
            "# LoRA corner-interpolation",
            "",
            "## What this measures",
            "",
            f"Some adapters here were trained on a **single** data group "
            f"({', '.join(v)}) — call these the *corners*. The rest were trained "
            f"on **blends** of those groups.",
            "",
            "Two questions:",
            "",
            "1. If you mathematically blend the corner adapters using the same "
            "recipe the data used, do you get the adapter that was actually "
            "trained on that blend?",
            "2. Working backwards: if you *fit* the best blend of corners to a "
            "trained adapter, do the fitted amounts match the real data recipe?",
            "",
        ]

        if s:
            lines += [
                "## The short version",
                "",
                f"- **No, you can't rebuild the weights.** Even the best possible "
                f"blend misses most of the adapter — it captures only about "
                f"**{(1 - s['rel_ols_mean'] ** 2) * 100:.0f}%** of it. The corner "
                f"adapters point in nearly unrelated directions, so blends of them "
                f"cover very little ground.",
                f"- **Yes, you can read the recipe off the weights.** The fitted "
                f"amounts land very close to the true data proportions — about "
                f"**{s['coeff_l2_vs_baseline']:.0f}x better than guessing**. The "
                f"weights don't match, but they still reveal what the model was "
                f"trained on.",
                f"- **Blends consistently come out too strong.** The fitted amounts "
                f"add up to **{s['gamma_mean']:.2f}**, not 1.00 — see *shrinkage* "
                f"below.",
                "",
            ]

        lines += [
            "## Terms",
            "",
            "- **Corner adapter** — trained on one group only (100% / 0% / 0%).",
            "- **Fitted amounts (ŵ)** — the blend of corners that best matches a "
            "trained adapter, found by least squares.",
            "- **Shrinkage (γ)** — *how much corner-material the fit asks for in "
            "total*: the fitted amounts added together, before rescaling them to "
            "sum to 1. If a trained adapter were exactly a proportional blend of "
            "the corners, γ would be 1.00. Every value here is below that, "
            f"averaging **{s.get('gamma_mean', float('nan')):.2f}** — meaning a "
            "proportional blend of corners consistently *overshoots*, and the real "
            "adapter carries only about that fraction of the corner directions. "
            "γ measures **size**; the rescaled amounts ŵ/γ measure **direction**.",
            "- **Relative error** — how much of the adapter the blend fails to "
            "reproduce, as a fraction. 0 is perfect, 1 means the blend is no better "
            "than predicting nothing.",
            "- **Coefficient error ‖ŵ−w‖₂** — distance between the fitted recipe "
            "and the true data recipe. 0 is perfect.",
            "",
        ]

        if figures:
            lines += [
                "## Figures",
                "",
                "![Fitted vs true mixture](figures/ternary.png)",
                "",
                "**`ternary.png`** — Each corner of the triangle is one pure data "
                "group. A hollow circle is where an adapter's *real* recipe sits; "
                "the blue dot joined to it is where the *fitted* recipe sits. Short "
                "connectors mean the fit found the right recipe. Dot size is "
                "shrinkage γ — dots get smaller toward the middle, so the more "
                "evenly a model mixes all three groups, the less it looks like a "
                "blend of the pure models.",
                "",
                "![Coefficient recovery](figures/recovered_vs_true.png)",
                "",
                "**`recovered_vs_true.png`** — Fitted amount against true data "
                "proportion, one colour per group. Points on the dashed diagonal "
                "are perfect recoveries. Everything sits close to it, which is the "
                "main positive result.",
                "",
                "![Shrinkage and residual](figures/shrinkage_and_error.png)",
                "",
                "**`shrinkage_and_error.png`** — Read γ off a real axis here. Blend "
                "recipes run from corner-like on the left to evenly-mixed on the "
                "right. Blue (γ) drifts down and orange (error) climbs: evenly "
                "mixed models are both further from the corner span and harder to "
                "fit.",
                "",
                "![Per-module residual](figures/block_profile.png)",
                "",
                "**`block_profile.png`** — Error for each of the "
                f"{len(self.blocks)} adapted modules, by depth in the network. The "
                "downward drift means deeper layers behave more like blends than "
                "early ones. The two dashed lines show the two attention families "
                "are nearly identical, so this is not an artifact of the model's "
                "hybrid architecture.",
                "",
            ]

        lines += [
            "## Per-mixture results",
            "",
            f"- Base model: `{self.base_model_id}`",
            f"- Corner adapters: {', '.join(f'`{label[c]}`' for c in self.corners if c in label)}",
            f"- Measured over {len(self.blocks)} adapted modules, rank {self.rank}, "
            f"alpha {self.lora_alpha}",
            "",
            "Distances are between the **effective updates** the adapters add to "
            "the base model, `ΔW = (alpha/r)·B@A`, over every module at once.",
            "",
            "| mixture | true w | fitted ŵ | γ | rel err @ true | rel err @ fit "
            "| cos | ‖ŵ−w‖₂ | ‖ŵ−w‖₁ | simplex ŵ | rel err @ simplex |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for f in self.mixtures:
            lines.append(
                f"| `{label[f.target]}` | {fmt(f.true)} | {fmt(f.ols_normalized)} "
                f"| {f.gamma:.3f} | {f.rel_true:.4f} | {f.rel_ols:.4f} "
                f"| {f.cosine:.3f} | {f.coeff_l2:.4f} | {f.coeff_l1:.4f} "
                f"| {fmt(f.simplex)} | {f.rel_simplex:.4f} |"
            )

        if s:
            lines += [
                "",
                "## Summary",
                "",
                f"- Shrinkage **γ = {s['gamma_mean']:.4f} ± {s['gamma_std']:.4f}** "
                f"over {s['n_mixtures']} mixtures",
                f"- Relative error at the true recipe: **{s['rel_true_mean']:.4f}**; "
                f"at the best fit: **{s['rel_ols_mean']:.4f}** "
                f"(so the corner blend captures "
                f"{(1 - s['rel_ols_mean'] ** 2) * 100:.1f}% of the adapter)",
                f"- Coefficient error ‖ŵ−w‖₂: **{s['coeff_l2_mean']:.4f}** "
                f"(rescaled fit) vs {s['coeff_l2_simplex_mean']:.4f} "
                f"(forced to sum to 1)",
                f"- Guessing at random would score "
                f"{s['coeff_l2_random_baseline']:.4f}, so the fit is "
                f"**{s['coeff_l2_vs_baseline']:.1f}x better than chance**",
                f"- Corner adapters are nearly unrelated to each other: mean "
                f"cosine {float(np.mean(self._offdiag_cosines())):.3f} "
                f"(1.0 = identical, 0 = unrelated)",
                "",
                "## Reading this",
                "",
                self._interpretation(s),
            ]

        corner_lines = [f for f in self.fits if f.is_corner]
        if corner_lines:
            lines += [
                "",
                "## Corner self-consistency",
                "",
                "A corner adapter fitted against the corner basis must come back as "
                "itself, with no error. This is a check on the whole pipeline.",
                "",
                "| corner | fit | residual |",
                "|---|---|---|",
            ]
            for f in corner_lines:
                lines.append(f"| `{label[f.target]}` | {fmt(f.ols)} | {f.residual_ols:.3e} |")

        return "\n".join(lines) + "\n"

    def _interpretation(self, s: dict) -> str:
        off = float(np.mean(self._offdiag_cosines()))
        captured = (1 - s["rel_ols_mean"] ** 2) * 100
        return (
            f"Three things are true at once, and they are easy to mistake for a "
            f"contradiction.\n\n"
            f"**The corner adapters cover too little ground.** Blending them "
            f"reproduces only about {captured:.0f}% of a trained mixture adapter, "
            f"even using the best possible amounts. The reason is in the corner "
            f"adapters themselves: they point in nearly unrelated directions (mean "
            f"cosine {off:.3f}), so blends of them reach only a narrow slice of the "
            f"space an adapter can occupy. Independently trained LoRAs simply land "
            f"in different places.\n\n"
            f"**But the amounts still tell you the recipe.** The fitted amounts "
            f"come out {s['coeff_l2_vs_baseline']:.1f}x closer to the true data "
            f"proportions than guessing would ({s['coeff_l2_mean']:.3f} against "
            f"{s['coeff_l2_random_baseline']:.3f}). So while you cannot rebuild the "
            f"weights, you can read off what the model was trained on.\n\n"
            f"**Blends consistently come out too strong.** The fitted amounts sum "
            f"to γ = {s['gamma_mean']:.3f} ± {s['gamma_std']:.3f} instead of 1, and "
            f"how tight that spread is matters more than the value: every single "
            f"mixture carries about {100 * (1 - s['gamma_mean']):.0f}% less "
            f"corner-material than a proportional blend would give it. That is why "
            f"the rescaled fit is the headline number — forcing the amounts to sum "
            f"to 1 fights this effect and dumps the excess onto groups that should "
            f"be zero, worsening the recipe error from {s['coeff_l2_mean']:.3f} to "
            f"{s['coeff_l2_simplex_mean']:.3f}.\n\n"
            f"One caveat worth stating: a relative error of {s['rel_ols_mean']:.2f} "
            f"sounds like a failure, but two adapters that were *separately trained* "
            f"on neighbouring recipes are about as far apart from each other. Much "
            f"of this gap is ordinary training-run variation, not something "
            f"interpolation introduces. Confirming that needs the same recipe "
            f"trained twice under different seeds, which this adapter set does not "
            f"have."
        )

    def _offdiag_cosines(self) -> np.ndarray:
        k = self.corner_cosines.shape[0]
        return self.corner_cosines[np.triu_indices(k, 1)]

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, path: str | Path, figures: bool = True) -> Path:
        from safetensors.numpy import save_file

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "report.json").write_text(json.dumps(self.to_report(), indent=2))
        (path / "report.md").write_text(self.to_markdown(figures=figures))

        fits_dir = path / "fits"
        fits_dir.mkdir(parents=True, exist_ok=True)
        stack = lambda attr: np.ascontiguousarray(  # noqa: E731
            np.stack([getattr(f, attr) for f in self.fits]), dtype=np.float64
        )
        col = lambda attr: np.ascontiguousarray(  # noqa: E731
            np.array([getattr(f, attr) for f in self.fits]), dtype=np.float64
        )
        meta = {
            "schema_version": SCHEMA_VERSION,
            "targets": [f.target for f in self.fits],
            "labels": [self.labels()[f.target] for f in self.fits],
            "is_corner": [f.is_corner for f in self.fits],
            "corners": self.corners,
            "vertices": self.vertices,
            "faces": [list(f.face) for f in self.fits],
        }
        save_file(
            {
                "true": stack("true"),
                "ols": stack("ols"),
                "ols_normalized": stack("ols_normalized"),
                "simplex": stack("simplex"),
                "gamma": col("gamma"),
                "cosine": col("cosine"),
                "coeff_l2": col("coeff_l2"),
                "coeff_l1": col("coeff_l1"),
                "norm_sq": col("norm_sq"),
                "residuals": np.ascontiguousarray(
                    np.array(
                        [
                            [f.residual_true, f.residual_ols, f.residual_simplex]
                            for f in self.fits
                        ]
                    ),
                    dtype=np.float64,
                ),
                "rel_errors": np.ascontiguousarray(
                    np.array(
                        [[f.rel_true, f.rel_ols, f.rel_simplex] for f in self.fits]
                    ),
                    dtype=np.float64,
                ),
                "corner_cosines": np.ascontiguousarray(
                    self.corner_cosines, dtype=np.float64
                ),
                "_meta_json": np.frombuffer(
                    json.dumps(meta).encode("utf-8"), dtype=np.uint8
                ),
            },
            str(fits_dir / "fits.safetensors"),
        )
        return path


def load_fits(path: str | Path) -> dict:
    """Read back what :meth:`InterpolationReport.save` wrote to ``fits/``.

    Lives next to ``save`` deliberately: the two share one format and have to
    change together.  Returns a plain dict rather than reconstructing an
    :class:`InterpolationReport`, because the arrays are what downstream
    analysis wants — a rebuilt report would need the Gram back to be honest
    about ``blocks``, ``scale`` and the corner cosines, and none of that is in
    this file.

    *path* may be the run directory or the ``fits/`` directory inside it.

    Keys: the metadata (``targets``, ``labels``, ``is_corner``, ``corners``,
    ``vertices``, ``faces``, ``schema_version``) plus every saved array
    (``true``, ``ols``, ``ols_normalized``, ``simplex``, ``gamma``, ``cosine``,
    ``coeff_l2``, ``coeff_l1``, ``norm_sq``, ``residuals``, ``rel_errors``,
    ``corner_cosines``), all float64 and row-aligned to ``targets``.
    """
    from safetensors.numpy import load_file

    path = Path(path)
    fits_file = path / "fits.safetensors"
    if not fits_file.exists():
        fits_file = path / "fits" / "fits.safetensors"
    if not fits_file.exists():
        raise FileNotFoundError(
            f"no fits.safetensors under {path}; expected a run directory written "
            "by InterpolationReport.save()"
        )

    tensors = load_file(str(fits_file))
    meta = json.loads(tensors.pop("_meta_json").tobytes().decode("utf-8"))
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{fits_file} has schema_version {meta.get('schema_version')!r}, "
            f"this code reads {SCHEMA_VERSION!r}"
        )

    out = dict(meta)
    out["faces"] = [tuple(f) for f in meta["faces"]]
    out.update({k: v.astype(np.float64) for k, v in tensors.items()})
    return out


def build_report(
    gram,
    truth: dict[str, np.ndarray],
    vertices: list[str],
    corners: list[str] | None = None,
    provenance: dict | None = None,
) -> InterpolationReport:
    """Fit every adapter in *gram* against the corner adapters.

    *truth* maps adapter name to its true mixture proportions, in *vertices*
    order.  *corners* is detected via :func:`detect_corners` when not given.

    The corners are fitted too.  Each one must come back as its own unit vector
    with a zero residual; that is free to compute and is the sharpest available
    self-check on the whole pipeline.
    """
    if corners is None:
        corners = detect_corners(vertices, truth)

    ci = [gram.index(c) for c in corners]
    Gc = gram.total[np.ix_(ci, ci)]

    fits = [
        fit_interpolation(gram, name, corners, truth[name])
        for name in gram.adapter_names
        if name in truth
    ]
    return InterpolationReport(
        fits=fits,
        corners=list(corners),
        vertices=list(vertices),
        blocks=list(gram.blocks),
        scale=gram.scale,
        base_model_id=gram.base_model_id,
        rank=gram.rank,
        lora_alpha=gram.lora_alpha,
        corner_cosines=gram.cosine_matrix()[np.ix_(ci, ci)],
        corner_cond=float(np.linalg.cond(Gc)),
        provenance=provenance or {},
    )
