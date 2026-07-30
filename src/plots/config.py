from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

#   Anchored to the repo base so paths resolve the same regardless of the working
#   directory a script or notebook happens to run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Figures output directory ──────────────────────────────────────────────────
GLOBAL_FIGURES_DIR = _REPO_ROOT / "figures"

# ── Colorblind-safe palette ───────────────────────────────────────────────────
PALETTE = sns.color_palette("colorblind", 10)

# ── Font registration (silent no-op if file not found) ───────────────────────
#   Checked in order; the first hit wins.  When none resolve, set_style() falls
#   back to DejaVu Sans (see "font.sans-serif" below).
#     MODEL_TAXONOMY_FONT  explicit override, wins when set
#     <repo>/fonts/        drop or symlink the .ttf here (gitignored)
#     per-user font dirs   macOS, then the XDG and legacy Linux locations
_FONT_PATHS = [
    os.environ.get("MODEL_TAXONOMY_FONT"),
    _REPO_ROOT / "fonts" / "LibreFranklin[wght].ttf",
    Path.home() / "Library/Fonts/LibreFranklin-VariableFont_wght.ttf",
    Path.home() / ".local/share/fonts/LibreFranklin[wght].ttf",
    Path.home() / ".fonts/LibreFranklin[wght].ttf",
]
_LIBRE_FRANKLIN_LOADED = False
for _fp in _FONT_PATHS:
    if _fp and os.path.exists(_fp):
        fm.fontManager.addfont(str(_fp))
        _LIBRE_FRANKLIN_LOADED = True
        # Variable fonts register at a single weight; suppress the harmless
        # "failed to find font weight X" warnings that follow from this.
        logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
        break

# ── Preset definitions ────────────────────────────────────────────────────────
#   figsize: (width, height) in inches
#   font_size: base pt — used for titles + axis labels
#   tick_size: font_size * 0.75 — used for tick labels + legend
#   axis_line: linewidth for axes, grids, ticks (all scaled from this)
#   save_dpi: dots per inch when saving
_PRESETS: dict[str, dict] = {
    "one_col": {
        "figsize": (6.5, 4.0),
        "font_size": 9,
        "axis_line": 1.0,
        "save_dpi": 300,
    },
    "two_col": {
        "figsize": (3.25, 2.5),
        "font_size": 8,
        "axis_line": 0.8,
        "save_dpi": 300,
    },
    "two_col_full": {
        "figsize": (6.5, 4.0),
        "font_size": 8,
        "axis_line": 0.8,
        "save_dpi": 300,
    },
    "poster": {
        "figsize": (8.0, 6.0),
        "font_size": 26,
        "axis_line": 3.0,
        "save_dpi": 150,
    },
    "slides": {
        "figsize": (6.0, 3.75),
        "font_size": 18,
        "axis_line": 1.5,
        "save_dpi": 150,
    },
}


def set_style(
    preset: str = "two_col",
    font_family: str = "sans-serif",
    fig_width: float | None = None,
    fig_height: float | None = None,
) -> tuple[float, float]:
    """Apply mnzk whitegrid style for the given output context.

    Returns (fig_width, fig_height) reflecting any overrides.
    preset: "one_col" | "two_col" | "two_col_full" | "poster" | "slides"
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown preset {preset!r}. Choose from: {list(_PRESETS)}")

    p = _PRESETS[preset]
    fs = p["font_size"]
    ts = fs * 0.75
    al = p["axis_line"]
    w, h = p["figsize"]
    w = fig_width if fig_width is not None else w
    h = fig_height if fig_height is not None else h

    mpl.rcdefaults()
    sns.set_style("whitegrid")

    mpl.rcParams.update({
        # ── Figure ──────────────────────────────────────────────────────────
        "figure.figsize": (w, h),
        "figure.dpi": 150,
        "figure.titlesize": fs,

        # ── Save ────────────────────────────────────────────────────────────
        "savefig.dpi": p["save_dpi"],
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # ── Font ────────────────────────────────────────────────────────────
        "font.family": font_family,
        "font.sans-serif": ["Libre Franklin", "DejaVu Sans"] if _LIBRE_FRANKLIN_LOADED else ["DejaVu Sans"],
        "font.serif": ["Libre Baskerville", "DejaVu Serif"],
        "font.size": fs,
        "text.usetex": False,

        # ── Axes labels + title ──────────────────────────────────────────────
        "axes.labelsize": fs,
        "axes.titlesize": fs,

        # ── Tick labels ──────────────────────────────────────────────────────
        "xtick.labelsize": ts,
        "ytick.labelsize": ts,

        # ── Legend ───────────────────────────────────────────────────────────
        "legend.fontsize": ts,
        "legend.title_fontsize": ts,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.8,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.2,
        "legend.framealpha": 0.8,

        # ── Lines + axes geometry ────────────────────────────────────────────
        "lines.linewidth": al,
        "axes.linewidth": al * 0.5,
        "grid.linewidth": al * 0.3,
        "xtick.major.width": al * 0.5,
        "ytick.major.width": al * 0.5,
        "xtick.major.size": al * 2,
        "ytick.major.size": al * 2,
    })

    return w, h


# ── Convenience formatter for linear axes ────────────────────────────────────
lin_formatter = ScalarFormatter(useMathText=True)
lin_formatter.set_scientific(True)
lin_formatter.set_powerlimits((-3, 3))
