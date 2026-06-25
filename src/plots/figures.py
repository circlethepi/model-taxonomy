from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
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

def _resolve_savepath(savepath: Any, title: str | None) -> Path:
    """Return a .png Path, always inside GLOBAL_FIGURES_DIR unless absolute."""
    if savepath is None:
        name = (title or "untitled").lower().replace(" ", "_")
        return GLOBAL_FIGURES_DIR / f"fig_{name}.png"
    p = Path(savepath).with_suffix(".png")
    if p.parent == Path("."):
        return GLOBAL_FIGURES_DIR / p
    return p


def _save(fig: plt.Figure, savepath: Path) -> None:
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath)


def _short_id(model_id: str) -> str:
    """Return the last path segment of a model/adapter ID."""
    return model_id.rstrip("/").split("/")[-1]


def _get_fig_ax(ax: plt.Axes | None, figsize=None) -> tuple[plt.Figure, plt.Axes]:
    if ax is not None:
        return ax.get_figure(), ax
    return plt.subplots(figsize=figsize)


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
    labels: list[str] | None = None,
    series: list[PlotSeries] | None = None,
    ax: plt.Axes | None = None,
    annotate: bool = False,
    marker_size: int = 100,
    title: str | None = None,
    savepath=None,
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
                                     cmap="viridis", marker="o", s=marker_size, zorder=3)
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

    _save(fig, _resolve_savepath(savepath, title or _title))
    return fig, ax


# ── plot_distance_heatmap ─────────────────────────────────────────────────────

def plot_distance_heatmap(
    dm,
    ax: plt.Axes | None = None,
    label_fn: Callable[[str], str] | None = None,
    title: str | None = None,
    fmt: str = ".2f",
    cmap: str = "viridis_r",
    annot: bool = True,
    colorbar: bool = True,
    cbar_ticks: list | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    tick_rotation: int = 0,
    savepath=None,
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
        cbar=colorbar,
        square=True,
    )

    if colorbar and cbar_ticks is not None:
        ax.collections[0].colorbar.set_ticks(cbar_ticks)

    if colorbar:
        cbar = ax.collections[0].colorbar
        tick_fs = plt.rcParams.get("xtick.labelsize", plt.rcParams["font.size"])
        cbar.ax.tick_params(labelsize=tick_fs)

    ax.set_xticklabels(ax.get_xticklabels(), rotation=tick_rotation)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    _title = title or f"{dm.taxonomy} | {dm.metric}"
    ax.set_title(_title)

    _save(fig, _resolve_savepath(savepath, title or _title))
    return fig, ax
