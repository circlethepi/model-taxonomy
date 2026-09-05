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
#
#   Brightened 2026-08-24 from the original muted set (#1F5FA9 / #C13B3B /
#   #E3A21A). The hues are unchanged — this is the same blue/red/gold tricolor,
#   pushed up in chroma and lightness. That matters more here than in a
#   categorical palette: 13 of the 16 models sit in the *interior* of the
#   simplex, and an interior point is a three-way blend, which is always less
#   saturated than any of its parents. Starting from muted anchors left the
#   interior nearly grey.
ANCHORS: dict[str, str] = {
    "g1": "#1E6FE8",   # blue
    "g2": "#F02B3A",   # red
    "g3": "#FFC220",   # gold
}

GROUP_TOPICS: dict[str, list[int]] = {
    "g1": [0, 6, 7, 9],
    "g2": [1, 3, 4],
    "g3": [2, 5, 8],
}

#: The whole run of ``NNNgI`` segments, matched **greedily**.
#:
#: This is the fix for a silent scoring bug, and the greed is the fix. The
#: pattern was ``_(\d{3})g1_(\d{3})g2_(\d{3})g3_``, which needs a trailing
#: underscore after ``g3`` -- and a four-group name supplies one, because ``g4``
#: follows it. So on a 4-group id it *matched*, dropped the fourth group and
#: returned a renormalized 3-vector, with no exception and no warning:
#:
#:     dolly_025g1_025g2_025g3_025g4  ->  an even 3-mix, truth is an even 4-mix
#:     oasst1_000g1_000g2_050g3_050g4 ->  PURE g3, truth is the g3-g4 midpoint
#:
#: The second is the damage: an edge midpoint scored as a pure vertex, producing
#: agreement numbers that look entirely plausible. Matching the run greedily and
#: then requiring the indices to be exactly ``1..K`` means a group can no longer
#: fall off the end -- there is nothing for the match to stop before.
_MIX_RUN_RE = re.compile(r"_((?:\d{3}g\d+_)+)")
_MIX_PART_RE = re.compile(r"(\d{3})g(\d+)_")


# ── Mixture parsing ───────────────────────────────────────────────────────────

def mixture_weights(model_id: str) -> tuple[float, ...]:
    """The ``K``-vector of group weights in an adapter id, normalized to sum to 1.

    ``K`` is whatever the id carries -- three for yahoo, four for dolly and
    oasst1 -- and is never assumed. A truncated vector here is not a plotting
    bug: ``simplex_suite.truth_weights`` parses these names to build the ground
    truth that every cross-level score, agreement table and Procrustes fit is
    measured against.

    Normalization is not cosmetic: the centre of a 3-group simplex is spelled
    ``033g1_033g2_033g3`` and its parts sum to 99, not 100, so using the raw
    integers would place it slightly off-centre and give it the wrong colour.
    """
    m = _MIX_RUN_RE.search(model_id)
    if m is None:
        raise ValueError(
            f"No mixture found in {model_id!r}; "
            "expected a '_NNNg1_NNNg2_..._NNNgK_' segment."
        )
    parts = _MIX_PART_RE.findall(m.group(1))
    indices = [int(i) for _, i in parts]
    if indices != list(range(1, len(parts) + 1)):
        raise ValueError(
            f"Mixture in {model_id!r} has group indices {indices}, "
            f"expected 1..{len(parts)} consecutively."
        )
    raw = np.array([int(p) for p, _ in parts], dtype=float)
    total = raw.sum()
    if total <= 0:
        raise ValueError(f"Mixture in {model_id!r} sums to {total}.")
    return tuple(raw / total)


def n_groups(model_ids: Sequence[str]) -> int:
    """The ``K`` these ids share, raising if they do not share one.

    Every array built from these weights is stacked -- ``truth_weights`` does an
    ``np.vstack`` -- so a mixed-width collection has to fail here rather than
    produce a ragged array or, worse, a silently padded one. Two datasets under
    one base model is exactly how a mixed list arises; see
    ``src.analysis.discovery.scan_cache``'s dataset filter.
    """
    widths = {len(mixture_weights(m)) for m in model_ids}
    if not widths:
        raise ValueError("no model ids to read a group count from")
    if len(widths) > 1:
        by_width = {
            k: sorted(m for m in model_ids if len(mixture_weights(m)) == k)[:3]
            for k in sorted(widths)
        }
        raise ValueError(
            f"model ids mix {sorted(widths)} groups, which cannot be stacked into "
            f"one weight array. Examples per width: {by_width}. This usually means "
            f"a cache holding two datasets was scanned without a dataset filter."
        )
    return widths.pop()


def mixture_label(model_id: str) -> str:
    """``'25/50/25'`` — a compact tick and annotation label.

    Widens with ``K``: a 4-group id reads ``'25/25/25/25'``.
    """
    w = mixture_weights(model_id)
    return "/".join(str(int(round(x * 100))) for x in w)


def sort_by_mixture(model_ids: Sequence[str]) -> list[str]:
    """Order ids by their weights descending, so vertices bracket the sequence.

    A stable, meaningful row order matters for the heatmaps: with an arbitrary
    order a block structure that exists is invisible.

    Raises on a mixed-width collection rather than ordering it -- the caller is
    about to stack these into a weight array, and a tuple of three sorts against
    a tuple of four without complaint.
    """
    n_groups(model_ids)
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


#: The three pure mixtures and the centre, as ``mixture_label`` spells them.
VERTEX_LABELS = ("100/0/0", "0/100/0", "0/0/100")
CENTRE_LABEL = "33/33/33"

#: Human-facing names for the three groups, for figures aimed at readers who do
#: not know the ``g1``/``g2``/``g3`` shorthand.
GROUP_DISPLAY = {"g1": "Group 1", "g2": "Group 2", "g3": "Group 3"}


def align_to_simplex(
    coords: np.ndarray,
    model_ids: Sequence[str],
    centre: str = CENTRE_LABEL,
    up: str = "100/0/0",
    right: str = "0/100/0",
) -> np.ndarray:
    """Put a 2-D embedding in the simplex's own frame.

    Translates *centre* to the origin and rotates *up* onto the positive y-axis.
    An MDS solution is determined only up to translation, rotation **and
    reflection**, so fixing the first two still leaves half the panels mirrored
    at random; *right* pins the third by requiring that vertex to land at
    ``x > 0``. Reflecting across the y-axis leaves *up* where it is, so the three
    constraints are independent and all three are satisfiable at once.

    The convention matches :func:`ternary_legend` — g1 at the top, g2 to the
    right, g3 to the left — so a scatter and the legend beside it can be read in
    the same orientation.

    This is a similarity transform: distances, stress and any barycentric
    projection are unchanged. It only chooses where to stand.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.shape[1] != 2:
        raise ValueError(f"align_to_simplex needs a 2-D embedding, got {coords.shape}")

    pos = {}
    for wanted in (centre, up, right):
        hits = [i for i, m in enumerate(model_ids) if mixture_label(m) == wanted]
        if not hits:
            raise ValueError(
                f"no model with mixture {wanted!r} in this embedding; the frame is "
                f"defined by {centre!r} (origin), {up!r} (+y) and {right!r} (+x side). "
                f"Present: {sorted({mixture_label(m) for m in model_ids})}"
            )
        pos[wanted] = hits[0]

    x = coords - coords[pos[centre]]

    v = x[pos[up]]
    if np.linalg.norm(v) < 1e-12:
        raise ValueError(
            f"{up!r} sits on top of {centre!r} in this embedding, so there is no "
            "direction to rotate onto the y-axis."
        )
    # Rotate by (pi/2 - theta) so `up` lands on +y.
    theta = np.arctan2(v[1], v[0])
    phi = np.pi / 2 - theta
    c, s = np.cos(phi), np.sin(phi)
    x = x @ np.array([[c, -s], [s, c]]).T

    if x[pos[right]][0] < 0:
        x = x * np.array([-1.0, 1.0])
    return x


def _in_simplex_frame(coords: np.ndarray, model_ids: Sequence[str]) -> np.ndarray:
    """:func:`align_to_simplex`, falling back to *coords* when it has no frame.

    Alignment is the default for every MDS scatter here, so it has to cope with
    a collection that does not define the frame — one with no pure-g1 model, or
    no centre — rather than taking the whole figure down over one panel. The
    frame is a convention for where to stand, and a collection that cannot say
    where to stand is still perfectly plottable from wherever MDS left it.

    The fallback is silent by design in only one direction: it fires on a
    *missing vertex*, which is a property of the collection and identical for
    every panel in the figure, so a partially-aligned grid is not reachable.
    """
    try:
        return align_to_simplex(coords, model_ids)
    except ValueError:
        return np.asarray(coords, dtype=float)


def ternary_legend(
    ax: plt.Axes,
    model_ids: Sequence[str] | None = None,
    anchors: Mapping[str, str] = ANCHORS,
    resolution: int = 220,
    marker_size: int = 26,
    label_models: bool = False,
    vertex_names: Mapping[str, str] | None = None,
    show_topics: bool = True,
    label_size: float = 5,
    vertex_size: float = 6,
    fontweight: str = "normal",
    fontfamily: str | None = None,
) -> plt.Axes:
    """Draw the filled simplex that the point colours are read from.

    This replaces a colourbar: the bar was legible only because composition used
    to be one number. With *model_ids* given, the sampled mixtures are marked, so
    the legend doubles as a map of which points the experiment actually visits.

    *vertex_names* renames the corners (``{"g1": "Group 1", ...}``); *show_topics*
    controls whether the topic-index list is printed under each one. Labels for
    the sampled mixtures are placed **radially outward from the centre** rather
    than at a fixed offset, which is what keeps sixteen of them legible on a
    triangle this small.
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

    text_kw = {"fontweight": fontweight}
    if fontfamily is not None:
        text_kw["fontfamily"] = fontfamily

    if model_ids:
        pts = _bary_to_xy(np.array([mixture_weights(m) for m in model_ids]))
        ax.scatter(pts[:, 0], pts[:, 1], s=marker_size, facecolors="none",
                   edgecolors="white", linewidths=0.9, zorder=3)
        if label_models:
            # Push each label away from the centroid along its own radius. A
            # fixed (3, 3) offset stacks the labels of the four points that share
            # a row of the grid; a radial one fans them out, and the centre point
            # (whose radius is zero) is nudged straight up.
            mid_xy = _bary_to_xy(np.array([1 / 3, 1 / 3, 1 / 3]))[0]
            for (x, y), mid in zip(pts, model_ids):
                d = np.array([x, y]) - mid_xy
                n = np.linalg.norm(d)
                dx, dy = (d / n * 9.5) if n > 1e-9 else (0.0, 8.0)
                ax.annotate(
                    mixture_label(mid), xy=(x, y), xytext=(dx, dy),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=label_size, color="0.15", zorder=4, **text_kw,
                )

    # Default to the `g1`/`g2`/`g3` shorthand: the dense surrogate x metric grids have
    # no room for "Group 1", and only the cross-taxonomy figure is aimed at a
    # reader who has not seen the shorthand. Pass GROUP_DISPLAY to rename.
    names = dict(vertex_names or {})
    # The three corners are also sampled mixtures, so when the points are
    # labelled the vertex name has to clear that label rather than land on top
    # of it. The radial offset above is 9.5pt, so the name goes beyond it.
    up, down = (26, -25) if (model_ids and label_models) else (13, -11)
    for (x, y), key, va, ha in [
        ((0.5, _SQRT3_2), "g1", "bottom", "center"),
        ((1.0, 0.0), "g2", "top", "left"),
        ((0.0, 0.0), "g3", "top", "right"),
    ]:
        label = names.get(key, key)
        if show_topics:
            label += "\n[" + ",".join(str(t) for t in GROUP_TOPICS[key]) + "]"
        ax.annotate(label, xy=(x, y),
                    xytext=(0, up) if va == "bottom" else (0, down),
                    textcoords="offset points", ha=ha, va=va,
                    fontsize=vertex_size, color="0.15", **text_kw)

    pad = 0.30 if (model_ids and label_models) else 0.22
    ax.set_xlim(-pad, 1.0 + pad)
    ax.set_ylim(-pad, _SQRT3_2 + pad * 0.9)
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

    Every panel is drawn in the simplex's own frame (:func:`align_to_simplex`):
    the centre mixture at the origin, pure g1 straight up, pure g2 to the right —
    the orientation :func:`ternary_legend` is drawn in. An MDS solution is fixed
    only up to translation, rotation and reflection, so without this each panel
    arrives in an arbitrary one of those and a grid of them cannot be compared by
    eye: two panels showing the *same* arrangement look unrelated because one is
    mirrored. It is a similarity transform, so no distance, stress or score
    moves — it only chooses where to stand.
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
                xy = _in_simplex_frame(geo.coordinates, geo.model_ids)
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


# ── Cross-taxonomy panel ──────────────────────────────────────────────────────

def crosslevel_mds(
    panels: Sequence[tuple[str, object, float] | tuple[str, object, float, float]],
    title: str,
    subtitle: str | None = None,
    savepath=None,
    anchors: Mapping[str, str] = ANCHORS,
    panel_w: float = 3.5,
    panel_h: float = 3.5,
    marker_size: int = 130,
    random_state: int = 0,
    label_points: Sequence[str] = VERTEX_LABELS + (CENTRE_LABEL,),
) -> plt.Figure:
    """One MDS panel per taxonomy, all in the simplex's own frame.

    *panels* is ``[(display_name, DistanceMatrix, dcor), ...]`` in the order they
    should appear, left to right, after the ternary legend — which sits **first**
    here rather than last, because it is the key the four panels are read
    through and a reader meets it before the data.

    A panel may carry a fourth element, the scaled residual Procrustes disparity
    against the ground truth
    (:func:`~src.analysis.ground_truth.disparity_vs_truth`), which is then named
    in the panel title between the dCor and the stress. It is passed in rather
    than computed here on purpose: this function fits its own MDS under
    *random_state*, and a caller that scored a different fit would have the
    figure and its own tables reporting two numbers for one configuration. Pass
    the disparity computed under this same seed, or leave it off.

    Two things separate this from :func:`mds_grid`, which stays as it is for the
    dense surrogate x metric grids. The shared frame is no longer one of them:
    :func:`align_to_simplex` is applied by both, and this function is where that
    convention started before it became the default everywhere.

    * **True 1:1 axes.** ``adjustable="box"`` with symmetric limits, rather than
      ``adjustable="datalim"``, so a unit of x is a unit of y *and* the panel is
      square. Under ``datalim`` matplotlib satisfies the aspect by stretching the
      data limits to fit whatever box the layout gives it, which is equal scaling
      in a non-square frame and reads as a distorted simplex.
    * **Four labels, not sixteen.** Only the three vertices and the centre are
      annotated. The interior mixtures are identified by colour, and the legend
      beside the panels is where they are named — repeating all sixteen labels in
      each of four panels is 64 pieces of text saying what the colours already
      say.

    Panel limits are **per panel**: the levels differ by roughly an order of
    magnitude in absolute MDS scale (structural ~0.6 against dataset ~0.1), so a
    shared limit would render three of the four as a dot at the origin. What is
    comparable across panels is the arrangement, not the size.
    """
    from src.analysis.bridge import fit_geometry
    from src.analysis.quality import kruskal_stress
    from src.plots.config import bold_capable_family

    # Libre Franklin is registered at a single weight (Thin), so `fontweight`
    # against it is a silent no-op — see `bold_capable_family`. Naming a family
    # that ships a real bold is the only way to get weight contrast here.
    family = bold_capable_family()
    bold = {"fontweight": "bold", "fontfamily": family}

    n = len(panels)
    fig = plt.figure(figsize=(panel_w * (n + 1) + 0.4, panel_h + 1.15),
                     layout="constrained")
    # The legend column is wider than a panel: the triangle carries sixteen
    # labels plus three vertex names, and it is the key everything else is read
    # through, so it should not be the smallest thing in the figure.
    gs = fig.add_gridspec(1, n + 1, width_ratios=[1.28] + [1.0] * n)

    lax = fig.add_subplot(gs[0, 0])
    ids = next((p[1].model_ids for p in panels), None)
    ternary_legend(lax, ids, anchors, label_models=True,
                   vertex_names=GROUP_DISPLAY, show_topics=False,
                   label_size=7.5, vertex_size=11, marker_size=34,
                   fontweight="bold", fontfamily=family)
    lax.set_title("Mixture key", fontsize=13, pad=10, **bold)

    keep = set(label_points)
    for k, panel in enumerate(panels):
        name, dm, dcor = panel[:3]
        procrustes = panel[3] if len(panel) > 3 else None
        ax = fig.add_subplot(gs[0, k + 1])
        geo = fit_geometry(dm, "mds", 2, random_state=random_state)
        xy = align_to_simplex(geo.coordinates, geo.model_ids)

        ax.axhline(0.0, color="0.88", lw=1.0, zorder=0)
        ax.axvline(0.0, color="0.88", lw=1.0, zorder=0)
        ax.scatter(xy[:, 0], xy[:, 1], c=model_colors(geo.model_ids, anchors),
                   s=marker_size, zorder=3, edgecolors="0.2", linewidths=1.0)

        for (x, y), mid in zip(xy, geo.model_ids):
            label = mixture_label(mid)
            if label in keep:
                ax.annotate(label, xy=(x, y), xytext=(0, 11),
                            textcoords="offset points", ha="center",
                            fontsize=9, color="0.1", zorder=4, **bold)

        # Three scores do not fit on one line of a 3.5" panel — at 13 pt they
        # run past the axes and collide with the neighbouring panel's title. So
        # the third one wraps, and the whole block drops a point. The two-score
        # form is left exactly as it was, on one line at 13 pt.
        stress = kruskal_stress(dm, geo)
        if procrustes is None:
            scores = f"dCor {dcor:.3f}  ·  stress {stress:.3f}"
            size = 13
        else:
            scores = (f"dCor {dcor:.3f}  ·  Procrustes {procrustes:.3f}"
                      f"\nstress {stress:.3f}")
            size = 12
        ax.set_title(f"{name}\n{scores}", fontsize=size, pad=10, **bold)

        # Symmetric about the origin, which the frame has already made the
        # centre mixture, so 1:1 scaling does not push the layout off-centre.
        r = float(np.abs(xy).max()) * 1.28
        ax.set_xlim(-r, r)
        ax.set_ylim(-r, r)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=8.5)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold")
            lbl.set_fontfamily(family)

    fig.suptitle(
        title + (f"\n{subtitle}" if subtitle else ""),
        fontsize=18, **bold,
    )
    if savepath is not None:
        save_figure(fig, str(savepath))
    return fig
