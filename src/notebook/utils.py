
from collections import Counter
import numpy as np

def sizes_to_alpha(sizes, min_alpha: float = 0.2, max_alpha: float = 1.0) -> list[float]:
    """Map dataset sizes to alpha values, linear in log-size space.

    The largest size maps to *max_alpha*; each equal multiplicative step in
    size corresponds to an equal additive step in alpha.
    """
    log_sizes = np.log(np.asarray(sizes, dtype=float))
    lo, hi = log_sizes.min(), log_sizes.max()
    if lo == hi:
        return [max_alpha] * len(sizes)
    t = (log_sizes - lo) / (hi - lo)
    return (min_alpha + (max_alpha - min_alpha) * t).tolist()


def sort_taxonomy(values):
    # values = [x.replace("dataset_embedding", "dataset") for x in values]
    order = ["recipe", "dataset_embedding", "structural", "functional", "behavioral"]
    return sorted(values, key=lambda x: order.index(x) if x in order else len(order))


def transform_geometry(coords, xflip=False, yflip=False, rotation=None):
    """transforms calculated coordinates"""

    coords = np.array(coords, dtype=float)
    if xflip:
        coords[:, 0] = -coords[:, 0]
    if yflip:
        coords[:, 1] = -coords[:, 1]
    if rotation is not None:
        angle = np.deg2rad(rotation)
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s], [s, c]])
        coords = coords @ R.T
    return coords

def multi_seed_heatmap_ticks(model_labels):
    """Gets tick locations and labels for heatmap plotting when multiple seeds
    are in use, eliminating redundancy"""
    
    label_counts = Counter(model_labels)

    labels, ticks = [], []
    loc = 0
    for key, val in label_counts.items():
        t = loc + np.ceil(val / 2)
        loc += val
        ticks.append(t)
        labels.append(key)
    
    return ticks, labels


def annotate_heatmap(ax, data, s=None, ticks=None, fmt=".2f", cmap=None, norm=None, **text_kw):
    """Annotate heatmap blocks with the block mean.
    If ticks are provided (from multi_seed_heatmap_ticks), block boundaries are inferred
    as midpoints between consecutive ticks. Otherwise uses fixed block size s (default 1).
    Text color is automatically white/black based on cell luminance; pass color= to override."""
    if ticks is not None:
        edges = np.round(
            np.concatenate([[0], (np.array(ticks[:-1]) + np.array(ticks[1:])) / 2, [data.shape[0]]])
        ).astype(int)
    else:
        if s is None:
            s = 1
        edges = np.arange(0, data.shape[0] + s, s)

    if "color" not in text_kw:
        mesh = ax.collections[0]
        _cmap = cmap if cmap is not None else mesh.get_cmap()
        _norm = norm if norm is not None else mesh.norm

    for r0, r1 in zip(edges[:-1], edges[1:]):
        for c0, c1 in zip(edges[:-1], edges[1:]):
            block = data[r0:r1, c0:c1]
            val = block.mean()
            kw = dict(text_kw)
            if "color" not in kw:
                r, g, b, _ = _cmap(_norm(val))
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                kw["color"] = "black" if lum > 0.5 else "white"
            ax.text((c0 + c1) / 2, (r0 + r1) / 2, format(val, fmt),
                    ha="center", va="center", **kw)


