from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

from .config import GLOBAL_FIGURES_DIR, PALETTE


# ── PlotSeries ────────────────────────────────────────────────────────────────

@dataclass
class PlotSeries:
    """Bundle of data + style for a single series in any plot type."""

    data: np.ndarray
    label: str | None = None
    color: Any | None = None       # None → cycle through PALETTE
    marker: str | None = None      # None → "o"
    linestyle: str | None = None   # None → "-"


def make_series(
    data: np.ndarray,
    label: str | None = None,
    color: Any | None = None,
    marker: str | None = None,
    linestyle: str | None = None,
) -> PlotSeries:
    """Convenience constructor for PlotSeries."""
    return PlotSeries(data=np.asarray(data), label=label, color=color,
                      marker=marker, linestyle=linestyle)


# ── Shared helpers ────────────────────────────────────────────────────────────

_IMAGE_SUFFIXES = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".tif", ".tiff"}


def _resolve_savepath(savepath: Any, title: str | None) -> Path:
    """Return an image Path, always inside GLOBAL_FIGURES_DIR unless absolute."""
    if savepath is None:
        name = (title or "untitled").lower().replace(" ", "_")
        return GLOBAL_FIGURES_DIR / f"fig_{name}.png"
    p = Path(savepath)
    if p.suffix.lower() not in _IMAGE_SUFFIXES:
        p = p.with_name(p.name + ".png")
    if p.parent == Path("."):
        return GLOBAL_FIGURES_DIR / p
    return p


def _save(fig: plt.Figure, savepath: Path) -> None:
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath)


def save_figure(fig: plt.Figure, savepath: Any = None, title: str | None = None) -> Path:
    """Save *fig* and return where it went.

    Bare filenames land in ``GLOBAL_FIGURES_DIR`` (the repo ``figures/``
    directory); paths with directories and absolute paths are used as given.
    With no *savepath*, the name is derived from *title*.
    """
    path = _resolve_savepath(savepath, title)
    _save(fig, path)
    return path


def _short_id(model_id: str) -> str:
    """Return the last path segment of a model/adapter ID."""
    return model_id.rstrip("/").split("/")[-1]


def _get_fig_ax(ax: plt.Axes | None, figsize=None) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.get_figure(), ax
    return plt.subplots(figsize=figsize)


# ─── Detail Helpers ───────────────────────────────────────────────────────────────

def nice_number(value, round_val=False):
    """Find a 'nice' number approximately equal to value"""
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    
    if round_val:
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
    else:
        if fraction <= 1:
            nice_fraction = 1
        elif fraction <= 2:
            nice_fraction = 2
        elif fraction <= 5:
            nice_fraction = 5
        else:
            nice_fraction = 10
    
    return nice_fraction * (10 ** exponent)


def nice_ticks(min_val, max_val, num_ticks=5):
    """Generate nice tick marks for a range"""
    range_val = nice_number(max_val - min_val, False)
    tick_spacing = nice_number(range_val / (num_ticks - 1), True)
    
    nice_min = math.floor(min_val / tick_spacing) * tick_spacing
    nice_max = math.ceil(max_val / tick_spacing) * tick_spacing
    
    ticks = []
    t = nice_min
    while t <= nice_max + tick_spacing * 0.5:
        ticks.append(round(t, 10))  # avoid float precision issues
        t += tick_spacing
    return ticks


def _add_colorbar(
    ax: plt.Axes,
    colormap: Union[str, plt.cm.Colormap],
    label: str | None = None,
    norm: plt.Normalize | None = None,
    ticks: list | None = None,
    ticklabels: list[str] | Callable | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Add a colorbar to an axis"""
    fig = ax.get_figure()

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])  # Required for ScalarMappable to work properly
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label)

    if ticks is not None:
        ticks = sorted(set(ticks))  # Ensure ticks are unique and sorted
        cbar.set_ticks(ticks)
    if ticklabels is not None:
        if ticklabels is not None and callable(ticklabels):
            ticklabels = [ticklabels(tick) for tick in ticks]
        cbar.set_ticklabels(ticklabels)

    return fig, ax




# ── plot_lines ────────────────────────────────────────────────────────────────

def plot_lines(
    x: np.ndarray,
    ys: np.ndarray | list[np.ndarray] | None = None,
    labels: list[str] | None = None,
    series: list[PlotSeries] | None = None,
    ax: plt.Axes | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    markers: bool = True,
    savepath=None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one or more line series on a shared axis.

    Provide either `series` (list of PlotSeries) or `ys` + optional `labels`.
    """
    fig, ax = _get_fig_ax(ax)
    x = np.asarray(x)

    if series is not None:
        for i, s in enumerate(series):
            color = s.color if s.color is not None else PALETTE[i % len(PALETTE)]
            mkr = s.marker if s.marker is not None else ("o" if markers else None)
            ls = s.linestyle if s.linestyle is not None else "-"
            ax.plot(x, np.asarray(s.data), color=color, marker=mkr,
                    linestyle=ls, label=s.label)
    elif ys is not None:
        ys_arr = np.atleast_2d(ys)
        for i, row in enumerate(ys_arr):
            lbl = labels[i] if labels is not None else None
            mkr = "o" if markers else None
            ax.plot(x, row, color=PALETTE[i % len(PALETTE)],
                    marker=mkr, label=lbl)
    else:
        raise ValueError("Provide either `series` or `ys`.")

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if labels is not None or series is not None:
        handles, lbls = ax.get_legend_handles_labels()
        if any(l is not None for l in lbls):
            ax.legend()

    _save(fig, _resolve_savepath(savepath, title))
    return fig, ax


# ── plot_scatter ──────────────────────────────────────────────────────────────

def plot_scatter(
    geometry,
    color_by: list | np.ndarray | None = None,
    colormap: str | plt.cm.Colormap = "plasma",
    labels: list[str] | None = None,
    series: list[PlotSeries] | None = None,
    ax: plt.Axes | None = None,
    annotate: bool = False,
    marker_size: int = 100,
    title: str | None = None,
    savepath=None,
    savefig=True
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot of a GeometryResult (2D MDS/UMAP/PCA coordinates).

    geometry: GeometryResult from src.core.geometry
    color_by: categorical list → PALETTE; numeric array → "viridis" colormap
    series: list of PlotSeries where each .data is shape (2,) for one point
    """
    fig, ax = _get_fig_ax(ax)

    if series is not None:
        for i, s in enumerate(series):
            xy = np.asarray(s.data).ravel()
            color = s.color if s.color is not None else PALETTE[i % len(PALETTE)]
            mkr = s.marker if s.marker is not None else "o"
            ax.scatter(xy[0], xy[1], color=color, marker=mkr, label=s.label, s=marker_size, zorder=3)
            if annotate and s.label:
                ax.annotate(s.label, xy=(xy[0], xy[1]),
                            xytext=(4, 4), textcoords="offset points")
    else:
        coords = geometry.coordinates
        ids = geometry.model_ids
        point_labels = labels if labels is not None else [_short_id(m) for m in ids]

        if color_by is not None:
            color_arr = np.asarray(color_by)
            if color_arr.dtype.kind in ("U", "S", "O"):
                unique = list(dict.fromkeys(color_arr))
                colors = [PALETTE[unique.index(v) % len(PALETTE)] for v in color_arr]
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=colors,
                                     marker="o", s=marker_size, zorder=3)
            else:
                scatter = ax.scatter(coords[:, 0], coords[:, 1], c=color_arr,
                                     cmap=colormap, marker="o", s=marker_size, zorder=3)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], color=PALETTE[0],
                       marker="o", s=marker_size, zorder=3)

        if annotate:
            for (xi, yi), lbl in zip(coords, point_labels):
                ax.annotate(lbl, xy=(xi, yi),
                            xytext=(4, 4), textcoords="offset points")

    _title = title or f"{geometry.taxonomy} ({geometry.method})"
    if getattr(geometry, "stress", None) is not None:
        _title += f"  [stress={geometry.stress:.3f}]"
    ax.set_title(_title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")

    if savefig:
        _save(fig, _resolve_savepath(savepath, title or _title))
    return fig, ax


# ── plot_distance_heatmap ─────────────────────────────────────────────────────

def plot_distance_heatmap(
    dm,
    ax: plt.Axes | None = None,
    label_fn: Callable[[str], str] | None = None,
    title: str | None = None,
    fmt: str = ".2f",
    cmap: str = "copper_r",
    annot: bool = True,
    colorbar: bool = True,
    cbar_ticks: list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    tick_rotation: int = 0,
    savepath=None,
    savefig=True,
) -> tuple[plt.Figure, plt.Axes]:
    """Heatmap of a DistanceMatrix.

    dm: DistanceMatrix from src.core.distance
    label_fn: maps model_id → display label (default: last "/" segment)
    cbar_ticks: explicit tick positions on the colorbar
    vmin/vmax: colormap limits (default: data range)
    tick_rotation: rotation in degrees for x-axis tick labels
    """
    _label = label_fn if label_fn is not None else _short_id
    tick_labels = [_label(m) for m in dm.model_ids]

    n = len(dm.model_ids)
    figsize = (0.8 * n + 1.5, 0.8 * n + 1.5)
    fig, ax = _get_fig_ax(ax, figsize=figsize)

    sns.heatmap(
        dm.matrix,
        ax=ax,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        annot=annot,
        fmt=fmt if annot else "",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        square=True,
    )

    if colorbar:
        norm = plt.Normalize(vmin=np.min(dm.matrix) if vmin is None else vmin,
                             vmax=np.max(dm.matrix) if vmax is None else vmax)
        fig, ax =_add_colorbar(ax, colormap=cmap, label="distance", norm=norm,
                      ticks=cbar_ticks)
    # if colorbar and cbar_ticks is not None:
    #     ax.collections[0].colorbar.set_ticks(cbar_ticks)

    # if colorbar:
    #     cbar = ax.collections[0].colorbar
        # tick_fs = plt.rcParams.get("xtick.labelsize", plt.rcParams["font.size"])
        # cax.tick_params(labelsize=tick_fs)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _title = title or f"{dm.taxonomy} | {dm.metric}"
    ax.set_title(_title)

    if savefig:
        _save(fig, _resolve_savepath(savepath, title or _title))
    return fig, ax



# ── encoding_legend ───────────────────────────────────────────────────────────

def encoding_legend(
    ax: plt.Axes,
    *groups: tuple[str, dict[str, dict]],
    loc: str | tuple | list = "best",
    **legend_kw,
) -> list[Any]:
    """Attach one legend per visual encoding, instead of one per line.

    A plot that encodes two variables at once — colour for one, linestyle or
    marker for the other — has as many lines as the product of the two, and a
    single legend listing every line makes the reader recover the encoding from
    the labels. Two short legends state it directly.

    Each *group* is ``(title, {label: line_kwargs})``, where ``line_kwargs`` are
    passed to ``matplotlib.lines.Line2D`` (``color``, ``linestyle``, ``marker``
    …). Entries carry no data; they are proxy handles drawn for the key alone.
    *loc* is either one location for every group — which overlaps them, so pass
    it only with a single group — or one location per group.

    Returns the legend artists, first group first. Only the last legend a call
    to ``ax.legend`` creates survives on the axes, so every group but the last
    is re-attached with ``add_artist``.
    """
    from matplotlib.lines import Line2D

    locs = list(loc) if isinstance(loc, list) else [loc] * len(groups)
    if len(locs) != len(groups):
        raise ValueError(f"{len(locs)} locations for {len(groups)} legend groups")

    legends = []
    for (title, entries), where in zip(groups, locs):
        handles = [Line2D([], [], label=label, **kw) for label, kw in entries.items()]
        legends.append(ax.legend(handles=handles, title=title, loc=where, **legend_kw))
    for leg in legends[:-1]:
        ax.add_artist(leg)
    return legends


# ── plot_radar ────────────────────────────────────────────────────────────────

def plot_radar(
    series: list[PlotSeries],
    axis_labels: list[str],
    ax: plt.Axes | None = None,
    start_angle: float = 90.0,
    direction: str = "ccw",
    rlim: tuple[float, float] | None = None,
    rticks: list[float] | None = None,
    fill_alpha: float = 0.12,
    marker_size: int = 5,
    label_pad: float = 0.0,
    spoke_axes: bool = False,
    tick_len: float = 0.018,
    rlabel_spoke: int | None = 0,
    rlabel_pad: float = 5.0,
    title: str | None = None,
    legend: bool = True,
    legend_kwargs: dict | None = None,
    savepath=None,
    savefig: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Radar (spider) plot: one closed polygon per series over shared spokes.

    Each ``PlotSeries.data`` is one value per spoke, in the same order as
    *axis_labels*.  Spokes are spaced evenly around the circle; *start_angle*
    (degrees, 0 = east, 90 = north) places the first one and *direction*
    ("ccw" or "cw") decides which way the rest follow, so the caller controls
    spoke placement purely by the order it passes *axis_labels*.

    *label_pad* pushes the spoke labels outward, in points.  On a full-range
    radar the polygon reaches the outer ring and its markers land on top of the
    labels, so a few points of padding is usually needed.

    *spoke_axes* swaps the default polar grid -- concentric rings plus a round
    outer frame -- for one straight axis line per spoke, each carrying tick
    marks at the *rticks* radii.  It reads as a set of shared axes radiating
    from the centre rather than as a web, which suits a radar whose spokes are
    unrelated quantities that merely share a scale.  *tick_len* sizes those
    marks as a fraction of the radial range, and *rlabel_spoke* indexes the one
    spoke that gets numeric tick labels (``None`` for none) -- repeating the
    numbers on every spoke only adds clutter, since the scale is shared.  Those
    labels are pushed *rlabel_pad* points to the side of their spoke, clear of
    the line and of the tick marks; the offset is perpendicular and measured on
    the page, so every label clears by the same amount whatever its radius.

    ``NaN`` values are drawn as gaps rather than pulling the polygon to zero.
    Passing *rlim* fixes the radial range, which is what makes two radars
    comparable; leaving it ``None`` autoscales each figure independently.
    """
    if direction not in ("ccw", "cw"):
        raise ValueError(f"direction must be 'ccw' or 'cw', got {direction!r}")

    n = len(axis_labels)
    if n < 3:
        raise ValueError(f"a radar needs at least 3 spokes, got {n}")

    if ax is None:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    else:
        if ax.name != "polar":
            raise ValueError("plot_radar needs an axes with projection='polar'")
        fig = ax.get_figure()

    #   The closing point repeats the first so each polygon joins up.
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    closed = np.concatenate([theta, theta[:1]])

    for i, s in enumerate(series):
        vals = np.asarray(s.data, dtype=float).ravel()
        if vals.size != n:
            raise ValueError(
                f"series {s.label!r} has {vals.size} values for {n} spokes"
            )
        vals = np.concatenate([vals, vals[:1]])
        color = s.color if s.color is not None else PALETTE[i % len(PALETTE)]
        ax.plot(closed, vals, color=color, label=s.label,
                marker=s.marker if s.marker is not None else "o",
                markersize=marker_size,
                linestyle=s.linestyle if s.linestyle is not None else "-",
                zorder=3)
        if fill_alpha:
            ax.fill(closed, vals, color=color, alpha=fill_alpha, zorder=2)

    ax.set_theta_offset(np.deg2rad(start_angle))
    ax.set_theta_direction(-1 if direction == "cw" else 1)
    ax.set_xticks(theta)
    ax.set_xticklabels(axis_labels)
    if label_pad:
        ax.tick_params(axis="x", pad=label_pad)
    if rlim is not None:
        ax.set_ylim(*rlim)
    if rticks is not None:
        ax.set_yticks(rticks)

    if spoke_axes:
        _draw_spoke_axes(ax, theta, tick_len, rlabel_spoke, rlabel_pad,
                         start_angle, direction)

    if title:
        ax.set_title(title)
    if legend and any(s.label for s in series):
        ax.legend(**(legend_kwargs or {"loc": "upper right",
                                       "bbox_to_anchor": (1.35, 1.12)}))

    if savefig:
        _save(fig, _resolve_savepath(savepath, title))
    return fig, ax


def _draw_spoke_axes(ax, theta, tick_len, rlabel_spoke, rlabel_pad,
                     start_angle, direction):
    """Replace a polar axes' rings and frame with one ticked axis per spoke.

    Called by :func:`plot_radar` for ``spoke_axes=True``.  Runs after the data
    and the limits are set, because every length here is measured against the
    final radial range.
    """
    lo, hi = ax.get_ylim()
    ticks = [t for t in ax.get_yticks() if lo <= t <= hi]
    #   Line colour and width follow whatever the active style gives the grid,
    #   so these hand-drawn axes still match the rest of the figure.
    color = plt.rcParams["grid.color"]
    lw = plt.rcParams["grid.linewidth"]

    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    for t in theta:
        ax.plot([t, t], [lo, hi], color=color, linewidth=lw,
                solid_capstyle="butt", zorder=1, clip_on=False)
        for r in ticks:
            #   A tick mark is a short arc across its spoke.  Its angular
            #   half-width has to shrink as the radius grows for every mark to
            #   come out the same length on the page; at r = 0 there is no
            #   angle that works, and no room for a mark either.
            if r <= lo:
                continue
            half = tick_len * (hi - lo) / r
            #   clip_on=False for the same reason as the spoke line: the mark at
            #   the outer limit straddles the axes boundary, so half of it is
            #   outside and would otherwise be cut away.
            ax.plot([t - half, t + half], [r, r], color=color, linewidth=lw,
                    solid_capstyle="butt", zorder=1, clip_on=False)

    if rlabel_spoke is None:
        ax.set_yticklabels([])
        return

    spoke = theta[rlabel_spoke % len(theta)]
    ax.set_rlabel_position(np.rad2deg(spoke))
    if not rlabel_pad:
        return

    #   set_rlabel_position centres the numbers on the spoke, where the axis
    #   line runs straight through them.  Shifting them along the spoke would
    #   only move them onto a different part of the same line, so the offset is
    #   perpendicular to it, in points, applied on top of whatever transform
    #   the tick labels already carry.
    on_page = np.deg2rad(start_angle) + (-1 if direction == "cw" else 1) * spoke
    dx, dy = np.sin(on_page), -np.cos(on_page)
    shift = mtransforms.ScaledTranslation(
        rlabel_pad * dx / 72, rlabel_pad * dy / 72,
        ax.get_figure().dpi_scale_trans)
    for lab in ax.get_yticklabels():
        lab.set_transform(lab.get_transform() + shift)
        #   Anchor the text on the side the shift came from, so the pad is a
        #   gap between line and glyphs rather than a nudge of the whole box.
        lab.set_horizontalalignment(
            "left" if dx > 0.1 else "right" if dx < -0.1 else "center")
        lab.set_verticalalignment(
            "bottom" if dy > 0.1 else "top" if dy < -0.1 else "center")


# ── plot_grouped_bars ─────────────────────────────────────────────────────────

def plot_grouped_bars(
    series: list[PlotSeries],
    group_labels: list[str],
    ax: plt.Axes | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    bar_width: float = 0.8,
    annotate: bool = False,
    annot_fmt: str = "{:.2f}",
    annot_size: float | None = None,
    annot_rotation: float = 0.0,
    annot_inside: bool | str = False,
    title: str | None = None,
    legend: bool = True,
    legend_kwargs: dict | None = None,
    savepath=None,
    savefig: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Grouped bar chart: one group per *group_labels* entry, one bar per series.

    Each ``PlotSeries.data`` holds one value per group, in the same order as
    *group_labels*.  *bar_width* is the fraction of a group's slot the bars
    together occupy, so the gap between groups stays the same however many
    series are drawn.  ``NaN`` values simply draw no bar.
    """
    n_groups = len(group_labels)
    n_series = len(series)
    if n_series == 0:
        raise ValueError("plot_grouped_bars needs at least one series")

    fig, ax = _get_fig_ax(ax)

    x = np.arange(n_groups, dtype=float)
    width = bar_width / n_series
    #   Offsets centre the whole cluster on the group's tick.
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * width

    #   Applied before anything is drawn: the "auto" annotation placement below
    #   measures each bar against the y-range, so the final range has to be in
    #   effect by then or short bars get labelled as though they were tall.
    if ylim is not None:
        ax.set_ylim(*ylim)

    for i, (s, off) in enumerate(zip(series, offsets)):
        vals = np.asarray(s.data, dtype=float).ravel()
        if vals.size != n_groups:
            raise ValueError(
                f"series {s.label!r} has {vals.size} values for {n_groups} groups"
            )
        color = s.color if s.color is not None else PALETTE[i % len(PALETTE)]
        bars = ax.bar(x + off, vals, width=width, color=color, label=s.label,
                      zorder=3)
        if annotate:
            for rect, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                inside = annot_inside
                if inside == "auto":
                    lo, hi = ax.get_ylim()
                    span = hi - lo
                    #   A quarter of the axis is enough height for a rotated
                    #   label; below that the text would spill past the base.
                    inside = span > 0 and (v - lo) / span > 0.25
                ax.annotate(annot_fmt.format(v),
                            xy=(rect.get_x() + rect.get_width() / 2, v),
                            xytext=(0, -3 if inside else 2),
                            textcoords="offset points",
                            ha="center", va="top" if inside else "bottom",
                            color="white" if inside else None,
                            fontsize=annot_size, rotation=annot_rotation,
                            zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_xlim(-0.5, n_groups - 0.5)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and any(s.label for s in series):
        ax.legend(**(legend_kwargs or {}))

    if savefig:
        _save(fig, _resolve_savepath(savepath, title))
    return fig, ax
