"""Colour system and panel grids for 3-group simplex experiments.

Earlier suites mixed *two* topic groups, so a model's composition was a scalar and
a 1-D ramp (``plasma`` keyed to ``% topic 0``) could carry it. The simplex3 suites
mix *three* groups, so composition is a point in a 2-simplex and no single ramp can
represent it without discarding an axis.

The replacement is a **barycentric blend**: a model's colour is its own mixture.
Three anchor hues sit at the pure vertices and every interior point is the
weight-average of them, mixed in Oklab so that equal weight steps look like equal
colour steps. The legend is therefore the simplex itself, drawn as a filled
triangle, rather than a bar.

Distance matrices keep the established ``copper_r``; these colours are deliberately
in a different part of colour space so the two figure types never read as the same
encoding.
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon

from src.plots.config import GLOBAL_FIGURES_DIR  # noqa: F401  (re-exported for callers)
from src.plots.figures import _add_colorbar, save_figure

# ── Anchors ───────────────────────────────────────────────────────────────────
#   Chosen to stay separable under deuteranopia and to share no hue family with
#   copper_r, so a distance matrix and an embedding can never be confused.
ANCHORS: dict[str, str] = {
    "g1": "#1F5FA9",   # deep blue
    "g2": "#C13B3B",   # crimson
    "g3": "#E3A21A",   # gold
}

GROUP_TOPICS: dict[str, list[int]] = {
    "g1": [0, 6, 7, 9],
    "g2": [1, 3, 4],
    "g3": [2, 5, 8],
}

_MIX_RE = re.compile(r"_(\d{3})g1_(\d{3})g2_(\d{3})g3_")


# ── Mixture parsing ───────────────────────────────────────────────────────────

def mixture_weights(model_id: str) -> tuple[float, float, float]:
    """``(w_g1, w_g2, w_g3)`` for an adapter id, normalized to sum to 1.

    Normalization is not cosmetic: the centre of the simplex is spelled
    ``033g1_033g2_033g3`` and its parts sum to 99, not 100, so using the raw
    integers would place it slightly off-centre and give it the wrong colour.
    """
    m = _MIX_RE.search(model_id)
    if m is None:
        raise ValueError(
            f"No 3-group mixture found in {model_id!r}; "
            "expected a '_NNNg1_NNNg2_NNNg3_' segment."
        )
    raw = np.array([int(m.group(i)) for i in (1, 2, 3)], dtype=float)
    total = raw.sum()
    if total <= 0:
        raise ValueError(f"Mixture in {model_id!r} sums to {total}.")
    return tuple(raw / total)


def mixture_label(model_id: str) -> str:
    """``'25/50/25'`` — a compact tick and annotation label."""
    w = mixture_weights(model_id)
    return "/".join(str(int(round(x * 100))) for x in w)


def sort_by_mixture(model_ids: Sequence[str]) -> list[str]:
    """Order ids by (g1, g2, g3) descending, so vertices bracket the sequence.

    A stable, meaningful row order matters for the heatmaps: with an arbitrary
    order a block structure that exists is invisible.
    """
    return sorted(model_ids, key=lambda m: tuple(-x for x in mixture_weights(m)))


# ── Oklab ─────────────────────────────────────────────────────────────────────
#   Björn Ottosson's Oklab. Blending in sRGB directly would darken and desaturate
#   mixtures (the classic muddy-midpoint problem); Oklab keeps a 50/50 blend
#   looking like it sits halfway between its parents.

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.maximum(c, 0.0) ** (1 / 2.4) - 0.055)


_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
_M2 = np.array([
    [0.2104542553,  0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050,  0.4505937099],
    [0.0259040371,  0.7827717662, -0.8086757660],
])
_M2_INV = np.linalg.inv(_M2)
_M1_INV = np.linalg.inv(_M1)


def rgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) sRGB in [0, 1] → (..., 3) Oklab."""
    lin = _srgb_to_linear(np.asarray(rgb, dtype=float))
    lms = lin @ _M1.T
    return np.cbrt(lms) @ _M2.T


def oklab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """(..., 3) Oklab → (..., 3) sRGB, clipped to [0, 1]."""
    lms = (np.asarray(lab, dtype=float) @ _M2_INV.T) ** 3
    lin = lms @ _M1_INV.T
    return np.clip(_linear_to_srgb(lin), 0.0, 1.0)


def _hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def oklab_delta_e(rgb_a, rgb_b) -> float:
    """Euclidean distance in Oklab — a perceptual difference, unlike RGB distance."""
    return float(np.linalg.norm(rgb_to_oklab(rgb_a) - rgb_to_oklab(rgb_b)))


# ── Barycentric colour ────────────────────────────────────────────────────────

def barycentric_color(
    weights: Sequence[float],
    anchors: Mapping[str, str] = ANCHORS,
) -> np.ndarray:
    """Blend the three anchors by *weights* in Oklab. Returns sRGB in [0, 1].

    A pure vertex reproduces its anchor exactly, so the legend and the points
    agree by construction rather than by eye.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,):
        raise ValueError(f"weights must have 3 entries, got {w.shape}")
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive value")
    w = w / w.sum()
    lab = np.stack([rgb_to_oklab(_hex_to_rgb(anchors[g])) for g in ("g1", "g2", "g3")])
    return oklab_to_rgb(w @ lab)


def model_colors(
    model_ids: Sequence[str],
    anchors: Mapping[str, str] = ANCHORS,
) -> list[np.ndarray]:
    """One barycentric colour per model id, in the given order."""
    return [barycentric_color(mixture_weights(m), anchors) for m in model_ids]


# ── Ternary legend ────────────────────────────────────────────────────────────

_SQRT3_2 = np.sqrt(3) / 2


def _bary_to_xy(w: np.ndarray) -> np.ndarray:
    """Barycentric weights → 2-D triangle coordinates (g1 top, g2 right, g3 left)."""
    w = np.atleast_2d(w)
    verts = np.array([[0.5, _SQRT3_2], [1.0, 0.0], [0.0, 0.0]])  # g1, g2, g3
    return w @ verts


def ternary_legend(
    ax: plt.Axes,
    model_ids: Sequence[str] | None = None,
    anchors: Mapping[str, str] = ANCHORS,
    resolution: int = 220,
    marker_size: int = 26,
    label_models: bool = False,
) -> plt.Axes:
    """Draw the filled simplex that the point colours are read from.

    This replaces a colourbar: the bar was legible only because composition used
    to be one number. With *model_ids* given, the sampled mixtures are marked, so
    the legend doubles as a map of which points the experiment actually visits.
    """
    # Fill: rasterize the triangle, colouring each pixel by its own barycentric
    # weights, then mask everything outside.
    xs = np.linspace(0.0, 1.0, resolution)
    ys = np.linspace(0.0, _SQRT3_2, resolution)
    X, Y = np.meshgrid(xs, ys)
    # invert the vertex map: solve for weights at each pixel
    w1 = Y / _SQRT3_2
    w2 = X - 0.5 * w1
    w3 = 1.0 - w1 - w2
    inside = (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)

    W = np.stack([w1, w2, w3], axis=-1)
    W = np.clip(W, 0.0, None)
    total = W.sum(axis=-1, keepdims=True)
    total[total == 0] = 1.0
    W = W / total

    lab = np.stack([rgb_to_oklab(_hex_to_rgb(anchors[g])) for g in ("g1", "g2", "g3")])
    img = oklab_to_rgb(W @ lab)
    rgba = np.concatenate([img, inside[..., None].astype(float)], axis=-1)

    ax.imshow(rgba, origin="lower", extent=(0.0, 1.0, 0.0, _SQRT3_2), interpolation="bilinear")
    ax.add_patch(Polygon(
        [[0.5, _SQRT3_2], [1.0, 0.0], [0.0, 0.0]],
        closed=True, fill=False, edgecolor="0.35", linewidth=0.8,
    ))

    if model_ids:
        pts = _bary_to_xy(np.array([mixture_weights(m) for m in model_ids]))
        ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, facecolors="none",
                   edgecolors="white", linewidths=0.9, zorder=3)
        if label_models:
            for (x, y), mid in zip(pts, model_ids):
                ax.annotate(mixture_label(mid), xy=(x, y), xytext=(3, 3),
                            textcoords="offset points", fontsize=5, color="0.2")

    for (x, y), key, va, ha in [
        ((0.5, _SQRT3_2), "g1", "bottom", "center"),
        ((1.0, 0.0), "g2", "top", "left"),
        ((0.0, 0.0), "g3", "top", "right"),
    ]:
        topics = ",".join(str(t) for t in GROUP_TOPICS[key])
        ax.annotate(f"{key}\n[{topics}]", xy=(x, y),
                    xytext=(0, 12) if va == "bottom" else (0, -10),
                    textcoords="offset points", ha=ha, va=va, fontsize=6, color="0.25")

    ax.set_xlim(-0.14, 1.14)
    ax.set_ylim(-0.16, _SQRT3_2 + 0.14)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ── Panel grids ───────────────────────────────────────────────────────────────
#   A cell is either a DistanceMatrix or a string explaining why it is absent.
#   Absence is common here and is information: CKA cannot run on a single-row
#   representation, and Bures-Wasserstein cannot span blocks of differing input
#   dim. Rendering the reason in place is honest; a blank panel would read as a
#   failure and a dropped column would hide the constraint.

Cell = object  # DistanceMatrix | str

#: Below this, a distance matrix carries no geometry worth embedding.
#:
#: Not machine epsilon, deliberately. Cosine and Frobenius over identical inputs
#: land at ~1e-16, but Bures-Wasserstein forms
#: ``d^2 = ||s_i||^2 + ||s_j||^2 - 2*nuclear(G)`` and then takes a square root, so
#: catastrophic cancellation puts identical inputs at ~sqrt(eps) ~ 1e-8. At a
#: 1e-12 threshold the BW panel of the embeddings control rendered a full scatter
#: with a 1e-7 axis — pure rounding noise, presented as though it were structure.
_DEGENERATE_TOL = 1e-6


def _blank(ax: plt.Axes, reason: str) -> None:
    ax.text(0.5, 0.5, reason, ha="center", va="center", fontsize=6,
            color="0.45", wrap=True, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def dm_grid(
    cells: Mapping[tuple[str, str], Cell],
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    savepath=None,
    label_fn: Callable[[str], str] = mixture_label,
    annot: bool = False,
    panel_w: float = 3.1,
    panel_h: float = 2.9,
) -> plt.Figure:
    """Distance-matrix grid: rows × metrics, ``copper_r``, one colourbar per panel.

    Per-panel colourbars rather than a shared one, because the metrics live on
    genuinely different scales — a shared scale would flatten three of the four
    columns into a single tone.
    """
    import seaborn as sns

    fig, axes = plt.subplots(
        len(rows), len(cols),
        figsize=(panel_w * len(cols), panel_h * len(rows)),
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            ax = axes[r][c]
            cell = cells.get((row, col))
            if cell is None or isinstance(cell, str):
                _blank(ax, cell or "not computed")
            elif float(np.max(np.abs(cell.matrix))) <= _DEGENERATE_TOL:
                # Same reasoning as in mds_grid: per-panel colour normalization
                # would stretch rounding noise across the full copper ramp and
                # draw a convincing pattern out of nothing.
                _blank(ax, "models identical\n"
                           f"(max distance {float(np.max(np.abs(cell.matrix))):.1e})")
            else:
                labels = [label_fn(m) for m in cell.model_ids]
                sns.heatmap(
                    cell.matrix, ax=ax, xticklabels=labels, yticklabels=labels,
                    annot=annot, fmt=".2f" if annot else "",
                    cmap="copper_r", cbar=False, square=True,
                )
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
                ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=5)
                _add_colorbar(
                    ax, colormap="copper_r", label="distance",
                    norm=plt.Normalize(float(np.min(cell.matrix)), float(np.max(cell.matrix))),
                )
            if r == 0:
                ax.set_title(col, fontsize=9)
            if c == 0:
                ax.set_ylabel(row, fontsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    if savepath is not None:
        save_figure(fig, str(savepath))
    return fig


def mds_grid(
    cells: Mapping[tuple[str, str], Cell],
    rows: Sequence[str],
    cols: Sequence[str],
    title: str,
    savepath=None,
    anchors: Mapping[str, str] = ANCHORS,
    annotate: bool = True,
    marker_size: int = 70,
    panel_w: float = 3.0,
    panel_h: float = 2.9,
    random_state: int = 0,
) -> plt.Figure:
    """MDS grid with barycentric point colours and a ternary legend panel.

    Panel titles carry Kruskal stress rather than sklearn's raw ``stress_``,
    which is in squared matrix units and so is not comparable between panels.
    """
    from src.analysis.bridge import fit_geometry
    from src.analysis.quality import kruskal_stress

    ncols = len(cols)
    fig = plt.figure(
        figsize=(panel_w * ncols + 1.9, panel_h * len(rows)),
        layout="constrained",
    )
    gs = fig.add_gridspec(len(rows), ncols + 1, width_ratios=[1.0] * ncols + [0.62])

    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            ax = fig.add_subplot(gs[r, c])
            cell = cells.get((row, col))
            if cell is None or isinstance(cell, str):
                _blank(ax, cell or "not computed")
            elif float(np.max(np.abs(cell.matrix))) <= _DEGENERATE_TOL:
                # No geometry to recover: sklearn's MDS divides by a zero
                # sum-of-squares, and whatever it returns comes from its random
                # init. Expected for the embeddings control, where the models
                # really are identical. The magnitude is printed rather than
                # asserted as zero, since each metric bottoms out differently.
                _blank(ax, "models identical\n"
                           f"(max distance {float(np.max(np.abs(cell.matrix))):.1e})")
            else:
                geo = fit_geometry(cell, "mds", 2, random_state=random_state)
                xy = geo.coordinates
                ax.scatter(
                    xy[:, 0], xy[:, 1],
                    c=model_colors(geo.model_ids, anchors),
                    s=marker_size, zorder=3, edgecolors="0.25", linewidths=0.5,
                )
                if annotate:
                    for (x, y), mid in zip(xy, geo.model_ids):
                        ax.annotate(mixture_label(mid), xy=(x, y), xytext=(3, 3),
                                    textcoords="offset points", fontsize=5)
                try:
                    stress = kruskal_stress(cell, geo)
                    ax.set_title(f"stress {stress:.3f}", fontsize=7)
                except Exception:
                    pass
                ax.set_aspect("equal", adjustable="datalim")
                ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(
                    f"{col}\n{ax.get_title()}" if ax.get_title() else col, fontsize=9
                )
            if c == 0:
                ax.set_ylabel(row, fontsize=7)

    lax = fig.add_subplot(gs[:, -1])
    ids = next(
        (cl.model_ids for cl in cells.values() if not isinstance(cl, (str, type(None)))),
        None,
    )
    ternary_legend(lax, ids, anchors)

    fig.suptitle(title, fontsize=12)
    if savepath is not None:
        save_figure(fig, str(savepath))
    return fig
