"""Figures for the corner-interpolation analysis.

Uses the repo's plotting conventions throughout — ``set_style`` for the rcParams
preset and ``PALETTE`` (seaborn's colorblind ramp) for series color, saved via
``save_figure``.

Color note: only the first three slots of ``PALETTE`` are used, and only ever
for series identity, never for magnitude.  Those three were checked against the
colorblind-safety criteria (OKLab ΔE under simulated protanopia and
deuteranopia, over *all* pairs rather than adjacent ones, since these are
scatter plots where any two series can end up side by side) and clear the
target: worst min(protan, deutan) ΔE = 9.2, worst normal-vision ΔE = 18.5.  Slot
1 (orange) sits slightly under the 3:1 contrast target against a white surface,
so every plot that uses it also carries direct labels rather than relying on the
legend swatch alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "make_all_figures",
    "make_geometry_figures",
    "per_block_rel_error",
    "plot_block_profile",
    "plot_procrustes_superposition",
    "plot_recovered_vs_true",
    "plot_shrinkage_and_error",
    "plot_ternary",
]


def _upright_vertices(k: int = 3) -> np.ndarray:
    """Simplex vertices from the repo helper, rotated so vertex 0 points up.

    ``ground_truth.simplex_vertices`` returns a regular simplex in an arbitrary
    rotation — correct, but a triangle sitting at a random angle is harder to
    read.  Rotation is an isometry, so this changes presentation only; every
    distance in the figure is the one the rest of the package would compute.
    """
    from src.analysis.ground_truth import simplex_vertices

    V = simplex_vertices(k)
    if k != 3:
        return V
    delta = np.pi / 2 - np.arctan2(V[0, 1], V[0, 0])
    c, s = np.cos(delta), np.sin(delta)
    return V @ np.array([[c, s], [-s, c]])   # row-vector rotation by +delta


def _pct_label(w: np.ndarray) -> str:
    """Compact mixture label, e.g. [0.25, 0.5, 0.25] -> "25/50/25"."""
    return "/".join(f"{100 * x:.0f}" for x in w)


def _project(weights: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Barycentric weights -> plane coordinates."""
    return np.asarray(weights, dtype=np.float64) @ V


def plot_ternary(report, savepath=None, title="Fitted coeffs vs true coeffs"):
    """The simplex, with each mixture's true point joined to its fitted point.

    Marker area encodes the shrinkage gamma, so the two independent findings —
    the direction is right, the magnitude is not — are visible in one panel.
    """
    import matplotlib.pyplot as plt

    from src.plots.config import PALETTE, set_style
    from src.plots.figures import save_figure

    set_style("one_col")
    fig, ax = plt.subplots()

    V = _upright_vertices(len(report.vertices))
    tri = np.vstack([V, V[:1]])
    ax.plot(tri[:, 0], tri[:, 1], color="0.75", lw=1.0, zorder=1)

    mixtures = report.mixtures
    truth = _project(np.stack([f.true for f in mixtures]), V)
    fitted = _project(np.stack([f.ols_normalized for f in mixtures]), V)
    gammas = np.array([f.gamma for f in mixtures])

    for (x0, y0), (x1, y1) in zip(truth, fitted):
        ax.plot([x0, x1], [y0, y1], color="0.55", lw=1.0, zorder=2)

    # Marker AREA is proportional to gamma, and the hollow "true" ring is drawn
    # at the area gamma = 1 would give. So a filled dot sitting inside its ring
    # *is* the shrinkage, read directly: area ratio = gamma, and the fitted dot
    # can never be larger than the ring because gamma < 1 throughout.
    #
    # An earlier version sized only the fitted dots and fixed the ring at an
    # unrelated size, which drew 11 of 13 dots larger than "true" for no reason
    # and read as though the coefficients exceeded 1. They never do.
    AREA_AT_GAMMA_1 = 110.0
    ax.scatter(
        truth[:, 0], truth[:, 1], s=AREA_AT_GAMMA_1, facecolors="none",
        edgecolors="0.35", linewidths=1.0, zorder=5,   # ring drawn ON TOP of the
        # filled dot: the pairs sit nearly concentric (that is the good result),
        # so the outline has to stay visible for the size comparison to read.
        label="true proportions (γ = 1)",
    )
    ax.scatter(
        fitted[:, 0], fitted[:, 1], s=AREA_AT_GAMMA_1 * gammas, c=[PALETTE[0]],
        edgecolors="white", linewidths=0.8, zorder=4,
        label="best fit (area ∝ γ)",
    )

    # Direct-label the vertices; the corner identity never rests on color.
    for i, name in enumerate(report.vertices):
        v = V[i] * 1.16
        ax.annotate(
            name, v, ha="center", va="center", fontweight="bold",
            color=PALETTE[i % 3],
        )
    # No per-point labels: a point's position in the triangle *is* its mixture,
    # so labelling all 13 would be redundant ink that collides with itself.

    ax.set_title(f"{title}\nShrinkage = γ = sum of unnormalized coefficients")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02), frameon=True)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.margins(0.14)
    return save_figure(fig, savepath, title)


def plot_recovered_vs_true(
    report, savepath=None, title="Normalized coefficient recovery error"
):
    """Fitted coefficient against true proportion, one series per component."""
    import matplotlib.pyplot as plt

    from src.plots.config import PALETTE, set_style
    from src.plots.figures import save_figure

    set_style("one_col")
    fig, ax = plt.subplots()

    mixtures = report.mixtures
    true = np.stack([f.true for f in mixtures])
    fit = np.stack([f.ols_normalized for f in mixtures])

    ax.plot([0, 1], [0, 1], color="0.7", lw=1.0, ls="--", zorder=1,
            label="perfect recovery")
    # Series identity is carried by the legend alone here (on-plot labels were
    # dropped by request). The report.md table remains the non-color-alone
    # fallback for the one palette slot that sits under the contrast target.
    for i, name in enumerate(report.vertices):
        ax.scatter(
            true[:, i], fit[:, i], s=34, color=PALETTE[i % 3],
            edgecolors="white", linewidths=0.8, zorder=3, label=name,
        )

    ax.set_xlabel("true data proportion")
    ax.set_ylabel("fitted coefficient (normalized)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    ax.set_xlim(-0.05, 1.08)
    ax.set_ylim(-0.05, 1.08)
    return save_figure(fig, savepath, title)


def plot_shrinkage_and_error(report, savepath=None, title="Shrinkage and residual"):
    """Gamma and relative error per mixture, ordered by interior-ness.

    Both quantities are dimensionless ratios in [0, 1], so they share one axis —
    a second y-scale would invent a comparison that the data does not support.
    """
    import matplotlib.pyplot as plt

    from src.plots.config import PALETTE, set_style
    from src.plots.figures import save_figure

    set_style("one_col")
    fig, ax = plt.subplots()

    mixtures = report.mixtures
    # "Interior-ness": distance from the nearest pure corner. Mixtures closest to
    # a corner sit left, the centroid furthest right.
    order = np.argsort([1.0 - float(np.max(f.true)) for f in mixtures])
    ordered = [mixtures[i] for i in order]
    x = np.arange(len(ordered))

    ax.plot(x, [f.gamma for f in ordered], marker="o", ms=5, lw=1.6,
            color=PALETTE[0], label="shrinkage γ = Σ fitted coeffs")
    ax.plot(x, [f.rel_ols for f in ordered], marker="s", ms=5, lw=1.6,
            color=PALETTE[1], label="relative error at best fit")

    ax.axhline(1.0, color="0.7", lw=1.0, ls="--", zorder=1)
    ax.annotate("γ = 1 (no shrinkage)", (len(x) - 1, 1.0),
                textcoords="offset points", xytext=(0, 5), ha="right",
                color="0.45", fontsize=plt.rcParams["xtick.labelsize"] * 0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([_pct_label(f.true) for f in ordered],
                       rotation=45, ha="right")
    ax.set_xlabel(
        f"mixture as {'/'.join(report.vertices)} %, "
        "ordered from corner-like to interior →"
    )
    ax.set_ylabel("ratio")
    s = report.summary()
    ax.set_title(f"{title}\nγ = {s['gamma_mean']:.3f} ± {s['gamma_std']:.3f}")
    # Both series are ratios read against the gamma = 1 reference, not lengths
    # read from zero, so the axis is focused on the occupied band.
    lo = min(min(f.gamma for f in ordered), min(f.rel_ols for f in ordered))
    ax.set_ylim(max(0.0, lo - 0.06), 1.06)
    ax.legend(loc="lower left", frameon=True)
    return save_figure(fig, savepath, title)


def per_block_rel_error(gram, report) -> tuple[np.ndarray, np.ndarray]:
    """Mean relative error at the best fit, computed block by block.

    Returns ``(rel_error, gamma)``, each ``(n_blocks,)``, averaged over the
    mixtures.  Cheap because the per-block Gram is already in memory — this is
    what keeping the block breakdown buys.
    """
    from src.interp.fitting import least_squares_weights, residual_sq

    ci = [gram.index(c) for c in report.corners]
    targets = [(gram.index(f.target), f) for f in report.mixtures]

    rel = np.zeros(len(gram.blocks))
    gam = np.zeros(len(gram.blocks))
    for b in range(len(gram.blocks)):
        G = gram.per_block[b]
        Gc = G[np.ix_(ci, ci)]
        rs, gs = [], []
        for t, _ in targets:
            g = G[t, ci]
            x = least_squares_weights(Gc, g)
            rs.append(np.sqrt(residual_sq(float(G[t, t]), g, Gc, x) / float(G[t, t])))
            gs.append(x.sum())
        rel[b] = float(np.mean(rs))
        gam[b] = float(np.mean(gs))
    return rel, gam


def plot_block_profile(report, gram, savepath=None, title="Per-module residual"):
    """Relative error for every module, by layer, split by attention family.

    Answers the question only a hybrid-attention model raises: is the failure to
    interpolate spread evenly through the stack, or concentrated in one kind of
    block?
    """
    import matplotlib.pyplot as plt

    from src.plots.config import PALETTE, set_style
    from src.plots.figures import save_figure

    set_style("one_col")
    fig, ax = plt.subplots()

    rel, _ = per_block_rel_error(gram, report)
    layers = np.array([l for l, _ in gram.blocks])
    modules = [m for _, m in gram.blocks]

    families: dict[str, list[int]] = {}
    for i, m in enumerate(modules):
        fam = "self_attn" if m in {"q_proj", "k_proj", "v_proj", "o_proj"} else "linear_attn"
        families.setdefault(fam, []).append(i)

    for k, (fam, idx) in enumerate(sorted(families.items())):
        color = PALETTE[k % 3]
        ax.scatter(layers[idx], rel[idx], s=26, color=color, alpha=0.85,
                   edgecolors="white", linewidths=0.6, zorder=3, label=fam)
        mean = float(np.mean(rel[idx]))
        ax.axhline(mean, color=color, lw=1.2, ls="--", zorder=2)
        # The family means sit within ~0.002 of each other, so their labels are
        # pushed apart vertically and anchored inside the axes.
        ax.annotate(
            f"{fam} mean {mean:.3f}",
            xy=(0.995, mean), xycoords=("axes fraction", "data"),
            textcoords="offset points", xytext=(0, 6 if k == 0 else -12),
            ha="right", color=color,
            fontsize=plt.rcParams["xtick.labelsize"] * 0.85, zorder=5,
        )

    ax.set_xlabel("layer index")
    ax.set_ylabel("relative error at best fit")
    ax.set_title(f"{title}\n{len(gram.blocks)} modules, averaged over "
                 f"{len(report.mixtures)} mixtures")
    ax.legend(loc="lower left", frameon=True)
    ax.margins(x=0.04)
    return save_figure(fig, savepath, title)


def make_all_figures(report, gram, output_dir) -> list[Path]:
    """Render every figure into *output_dir*, returning the paths written."""
    import matplotlib

    matplotlib.use("Agg")  # headless: these run on a login node with no display

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Absolute paths: save_figure sends bare filenames to the repo-global
    # figures/ directory, which is not where a results tree wants them.
    return [
        plot_ternary(report, savepath=out.resolve() / "ternary.png"),
        plot_recovered_vs_true(report, savepath=out.resolve() / "recovered_vs_true.png"),
        plot_shrinkage_and_error(report, savepath=out.resolve() / "shrinkage_and_error.png"),
        plot_block_profile(report, gram, savepath=out.resolve() / "block_profile.png"),
    ]


def plot_procrustes_superposition(
    cg_report, savepath=None, title="Coefficient geometry vs the true simplex"
):
    """The two arrangements laid on top of each other, one panel per estimate.

    Procrustes reports a single number for a whole configuration, which cannot
    distinguish "everything is slightly off" from "one adapter is badly wrong
    and the rest are perfect".  Drawing the superposition it already computed —
    ``aligned_a`` is the truth, ``aligned_b`` the fit, both in the fitted frame —
    puts that distinction on the page: each connector is one adapter's residual,
    the same quantity ``per_point_residuals`` returns.

    Axis limits are shared across panels so the panels are comparable to each
    other, not just internally.  Coordinates are post-superposition, so the axes
    carry no units worth labelling and are left bare.
    """
    import matplotlib.pyplot as plt

    from src.plots.config import PALETTE, set_style
    from src.plots.figures import save_figure

    kinds = [k for k in ("ols_normalized", "simplex") if k in cg_report.results]
    set_style("two_col")
    # Square-ish panels: the superposed clouds have no preferred axis, and the
    # equal aspect ratio below would waste any extra width as margin.
    fig, axes = plt.subplots(
        1, len(kinds), figsize=(3.5 * len(kinds), 4.0),
        squeeze=False, layout="constrained",
    )
    axes = axes[0]

    truth_color, fit_color = PALETTE[7], PALETTE[0]
    label = cg_report.labels
    corner = list(cg_report.is_corner)

    spans = []
    for ax, kind in zip(axes, kinds):
        r = cg_report.results[kind]
        A = np.asarray(r.procrustes.aligned_a.coordinates, dtype=np.float64)
        B = np.asarray(r.procrustes.aligned_b.coordinates, dtype=np.float64)
        ids = r.model_ids

        for i in range(len(ids)):
            ax.plot(
                [A[i, 0], B[i, 0]], [A[i, 1], B[i, 1]],
                color="0.45", lw=1.0, zorder=1, solid_capstyle="round",
            )
        # Corners are drawn as squares: they fit themselves exactly, so their
        # near-zero residuals are a property of the construction rather than a
        # result, and should not read as evidence.
        for is_c, marker, size in ((False, "o", 42), (True, "s", 50)):
            sel = [i for i in range(len(ids)) if corner[i] == is_c]
            if not sel:
                continue
            ax.scatter(
                A[sel, 0], A[sel, 1], s=size, marker=marker, facecolors="none",
                edgecolors=truth_color, linewidths=1.4, zorder=3,
                label="true recipe" if not is_c else None,
            )
            ax.scatter(
                B[sel, 0], B[sel, 1], s=size * 0.76, marker=marker,
                color=fit_color, zorder=4,
                label="fitted coefficients" if not is_c else None,
            )

        ax.set_title(f"{kind}\ndisparity {r.procrustes.disparity:.4f}", pad=8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal", adjustable="box")
        spans.append((np.vstack([A, B]).min(0), np.vstack([A, B]).max(0)))

    lo = np.min([s[0] for s in spans], axis=0)
    hi = np.max([s[1] for s in spans], axis=0)
    # One square window around both panels, so the two are comparable to each
    # other and not only internally.
    center, half = (lo + hi) / 2, float(np.max(hi - lo)) / 2
    half *= 1.18
    for ax, kind in zip(axes, kinds):
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)

        # Name the worst-fitting adapter, flipping the label to the inward side
        # so it cannot run off the panel.
        r = cg_report.results[kind]
        worst = int(np.argmax(r.residuals))
        B = np.asarray(r.procrustes.aligned_b.coordinates, dtype=np.float64)
        right = B[worst, 0] > center[0]
        ax.annotate(
            label.get(r.model_ids[worst], r.model_ids[worst]),
            xy=(B[worst, 0], B[worst, 1]), textcoords="offset points",
            xytext=(-7 if right else 7, 6), ha="right" if right else "left",
            fontsize=plt.rcParams["xtick.labelsize"] * 0.8,
            color=fit_color, zorder=5,
        )

    axes[0].legend(loc="upper left", frameon=True)
    n = len(cg_report.model_ids)
    n_corner = int(sum(corner))
    subtitle = f"{n} adapters" + (
        f", {n_corner} corners (squares)" if n_corner else ", mixtures only"
    )
    fig.suptitle(f"{title}\n{subtitle} — aligned; each connector is one residual")
    return save_figure(fig, savepath, title)


def make_geometry_figures(cg_report, output_dir) -> list[Path]:
    """Render the coefficient-geometry figures into *output_dir*."""
    import matplotlib

    matplotlib.use("Agg")  # headless: these run on a login node with no display

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return [
        plot_procrustes_superposition(
            cg_report, savepath=out.resolve() / "procrustes_superposition.png"
        )
    ]
