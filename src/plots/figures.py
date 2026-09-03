from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
import math

import matplotlib.pyplot as plt
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
