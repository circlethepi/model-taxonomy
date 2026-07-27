from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from src.notebook.lora_weights import _PROJ_LONG


def model_label(name: str) -> str:
    """Extract topic-0 percentage label from an adapter name.

    E.g. 'yahoo_75t0_25t1_n1000_s42_r16' → '75'.
    """
    m = re.search(r"(\d+)t0[_\-](\d+)t1", name)
    if m:
        n0, n1 = int(m.group(1)), int(m.group(2))
        return str(round(100 * n0 / (n0 + n1)))
    return name


def _normalize_layers(weights, layers: int | list[int] | None) -> list[int]:
    """Resolve a layer arg to a sorted list; bare int/None both supported."""
    if layers is None:
        return sorted(weights.layers)
    if isinstance(layers, int):
        return [layers]
    return sorted(layers)


def _normalize_projections(weights, projections: str | list[str] | None) -> list[str]:
    """Resolve a projections arg to a sorted list of short-form names."""
    if projections is None:
        return sorted(weights.projections)
    raw = [projections] if isinstance(projections, str) else list(projections)
    return sorted({_PROJ_LONG.get(p.lower(), p.lower()) for p in raw})


# ---------------------------------------------------------------------------
# Procrustes alignment helpers
# ---------------------------------------------------------------------------

def _frob_sq(A: np.ndarray, B: np.ndarray) -> float:
    """||B @ A||_F^2 in r×r space (A: r×d, B: d×r).

    tr(A^T B^T B A) = tr(B^T B  A A^T) via cyclic property — all r×r ops.
    """
    return float(np.trace((B.T @ B) @ (A @ A.T)))


def _procrustes_svd(
    A_i: np.ndarray, B_i: np.ndarray,
    A_j: np.ndarray, B_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Low-rank Procrustes: find Q* = argmin_Q ||P_i Q - P_j||_F, Q^TQ = I.

    Avoids forming the d×d product P_i^T P_j by working entirely in r×r space:

      K    = B_i^T B_j                    (r×r)
      SVD  K → U_K, s_K, V_K
      F    = A_i^T U_K,  QR → Q_F, R_F   (d×r, r×r)
      G    = A_j^T V_K,  QR → Q_G, R_G   (d×r, r×r)
      SVD  R_F diag(s_K) R_G^T → U_s σ V_s^T   (r×r)
      U_ij = Q_F U_s,  V_ij = Q_G V_s    (d×r)

    Returns
    -------
    U_ij  : (d, r) left  singular vectors of P_i^T P_j
    V_ij  : (d, r) right singular vectors of P_i^T P_j
    sigma : (r,)  singular values of P_i^T P_j (Q* = U_ij @ V_ij^T)
    tr_i  : ||P_i||_F^2
    tr_j  : ||P_j||_F^2
    """
    K = B_i.T @ B_j                                  # (r, r)
    U_K, s_K, Vk_T = np.linalg.svd(K)
    V_K = Vk_T.T                                      # (r, r)

    F = A_i.T @ U_K                                   # (d, r)
    G = A_j.T @ V_K                                   # (d, r)

    Q_F, R_F = np.linalg.qr(F)                       # Q_F (d,r), R_F (r,r)
    Q_G, R_G = np.linalg.qr(G)                       # Q_G (d,r), R_G (r,r)

    M_inner = R_F @ np.diag(s_K) @ R_G.T             # (r, r)
    U_s, sigma, Vs_T = np.linalg.svd(M_inner)
    V_s = Vs_T.T                                      # (r, r)

    return (
        Q_F @ U_s,               # U_ij (d, r)
        Q_G @ V_s,               # V_ij (d, r)
        sigma,                   # (r,)
        _frob_sq(A_i, B_i),
        _frob_sq(A_j, B_j),
    )


def _load_or_compute_alignment(
    name_i: str, name_j: str,
    A_i: np.ndarray, B_i: np.ndarray,
    A_j: np.ndarray, B_j: np.ndarray,
    layer: int, proj: str,
    cache_dir: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return (U_ij, V_ij, sigma, tr_i, tr_j) aligning P_i onto P_j.

    Cache files are keyed by (layer, proj, canonical_i, canonical_j) where
    canonical_i < canonical_j lexicographically.  The stored alignment always
    maps the lex-first adapter; for the reverse pair U and V are swapped on
    return.

    If cache_dir is None, the result is computed but not saved.
    """
    canonical_i, canonical_j = sorted([name_i, name_j])
    swapped = name_i != canonical_i  # name_i is the lex-larger one

    loaded = False
    if cache_dir is not None:
        cache_path = cache_dir / str(layer) / proj / f"{canonical_i}__{canonical_j}.npz"
        if cache_path.exists():
            d = np.load(cache_path)
            U_can, V_can = d["U"], d["V"]
            sigma = d["sigma"]
            tr_can_i, tr_can_j = float(d["tr_i"]), float(d["tr_j"])
            loaded = True

    if not loaded:
        # Compute in canonical order (aligns P_{can_i} → P_{can_j})
        if swapped:
            # name_i = can_j, name_j = can_i  →  canonical args are (A_j, B_j, A_i, B_i)
            U_can, V_can, sigma, tr_can_i, tr_can_j = _procrustes_svd(
                A_j, B_j, A_i, B_i
            )
        else:
            U_can, V_can, sigma, tr_can_i, tr_can_j = _procrustes_svd(
                A_i, B_i, A_j, B_j
            )
        if cache_dir is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(cache_path),
                U=U_can, V=V_can, sigma=sigma,
                tr_i=np.float64(tr_can_i), tr_j=np.float64(tr_can_j),
            )

    if swapped:
        # Caller wants P_{name_i}=P_{can_j} aligned to P_{name_j}=P_{can_i}
        # Reverse Procrustes: swap U↔V and tr_i↔tr_j
        return V_can, U_can, sigma, tr_can_j, tr_can_i
    return U_can, V_can, sigma, tr_can_i, tr_can_j


# ---------------------------------------------------------------------------
# Frobenius distance matrix
# ---------------------------------------------------------------------------

def frobenius_distance_matrix(
    weights,
    layers: int | list[int] | None = None,
    projections: str | list[str] | None = None,
    align: bool = False,
    cache_dir: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pairwise Frobenius distance between concatenated LoRA products B @ A.

    Uses a low-rank formulation throughout (no d×d matrices formed), summed
    over every selected (layer, proj) block:

      ||v_i - v_j||_2^2 = Σ_{layer,proj} ||P_i - P_j||_F^2
                         = Σ tr_i + Σ tr_j - 2·Σ tr(B_i^T B_j · A_j A_i^T)   (all r×r)

    where v_i is the vector obtained by raveling and concatenating each
    B @ A product across the selected layers/projections — see
    docs/notes/frobenius_bw_generalization.md for the proof that this equals
    the true Frobenius/Euclidean distance between the concatenated vectors.

    With ``align=True`` the Procrustes-aligned distance is returned instead;
    this is only supported for a single (layer, proj) block (multi-block
    Procrustes alignment is not implemented — raises ``ValueError``):

      ||P_i Q* - P_j||_F^2 = tr_i + tr_j - 2·sum(σ_ij)

    where σ_ij are the singular values of P_i^T P_j (computed in r×r space).
    Alignment matrices are loaded from / saved to ``cache_dir`` if provided.

    Parameters
    ----------
    weights : LoRAWeightCollection
    layers : layer indices to include; default is all layers loaded in *weights*
    projections : projections to include, e.g. ``"o"`` or ``["k", "v"]``;
        accepts short (``"o"``) or long (``"o_proj"``) forms.
        Default is all projections loaded in *weights*.
    align : whether to apply Procrustes alignment before computing the norm
        (single block only)
    cache_dir : directory for caching alignment matrices (None → no cache)
    """
    names = weights.keys()
    n = len(names)

    _layers = _normalize_layers(weights, layers)
    _projs = _normalize_projections(weights, projections)
    blocks = [(layer, proj) for layer in _layers for proj in _projs]

    if align and len(blocks) != 1:
        raise ValueError(
            f"align=True is only supported for a single (layer, projection) "
            f"block; got {len(blocks)} blocks. Multi-block Procrustes "
            f"alignment is not implemented."
        )

    # Precompute per-adapter, per-block quantities.
    As: list[list[np.ndarray]] = []
    Bs: list[list[np.ndarray]] = []
    trs: list[float] = []
    for name in tqdm(names, desc="frobenius precompute"):
        adapter_As: list[np.ndarray] = []
        adapter_Bs: list[np.ndarray] = []
        total = 0.0
        for layer, proj in blocks:
            A = weights[name].matrix(layer, proj, "A").astype(np.float64)
            B = weights[name].matrix(layer, proj, "B").astype(np.float64)
            adapter_As.append(A)
            adapter_Bs.append(B)
            total += _frob_sq(A, B)
        As.append(adapter_As)
        Bs.append(adapter_Bs)
        trs.append(total)

    D = np.zeros((n, n))
    pairs = list(combinations(range(n), 2))
    desc = "frobenius (aligned)" if align else "frobenius"
    for i, j in tqdm(pairs, desc=desc, total=len(pairs)):
        if align:
            layer, proj = blocks[0]
            _, _, sigma, _, _ = _load_or_compute_alignment(
                names[i], names[j], As[i][0], Bs[i][0], As[j][0], Bs[j][0],
                layer, proj, cache_dir,
            )
            d2 = trs[i] + trs[j] - 2.0 * float(np.sum(sigma))
        else:
            cross = 0.0
            for k in range(len(blocks)):
                K_ij = Bs[i][k].T @ Bs[j][k]                    # (r, r)
                cross += float(np.trace(K_ij @ (As[j][k] @ As[i][k].T)))
            d2 = trs[i] + trs[j] - 2.0 * cross
        D[i, j] = D[j, i] = float(np.sqrt(max(d2, 0.0)))
    return list(names), D


# ---------------------------------------------------------------------------
# CKA distance matrix  (low-rank, stays in rank-space throughout)
# ---------------------------------------------------------------------------

def cka_distance_matrix(
    weights,
    layer: int,
    proj: str,
    align: bool = False,
    cache_dir: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pairwise linear CKA distance (1 - CKA) between LoRA products.

    Uses a low-rank reformulation that avoids forming the full d×d kernel
    matrices.  For P_i = B_i A_i (B_i: d×r, A_i: r×d):

      K_i = P_i P_i^T  (centered gram)
      HSIC(K_i, K_j) = tr(C_j M_ij C_i M_ij^T) / (d-1)^2

    where C_i = A_i A_i^T (r×r), B_ic = col-centered B_i, M_ij = B_jc^T B_ic (r×r).

    Note on ``align``:
        Linear CKA is invariant to orthogonal transformations —
        K(P Q) = P Q Q^T P^T = P P^T = K(P) — so alignment has no effect
        on the result.  The parameter is accepted for API consistency.

    Parameters
    ----------
    weights : LoRAWeightCollection
    layer, proj : layer index and projection short name
    align : accepted but has no effect (CKA is orthogonally invariant)
    cache_dir : unused (kept for API consistency)
    """
    names = weights.keys()
    n = len(names)

    # Precompute per-adapter quantities in rank space.
    C: list[np.ndarray] = []   # A A^T  (r×r)
    Bc: list[np.ndarray] = []  # col-centered B  (d×r)
    for name in tqdm(names, desc="cka precompute"):
        A = weights[name].matrix(layer, proj, "A").astype(np.float64)
        B = weights[name].matrix(layer, proj, "B").astype(np.float64)
        C.append(A @ A.T)
        Bc.append(B - B.mean(axis=0, keepdims=True))

    n_rows = Bc[0].shape[0]  # d

    def _hsic(i: int, j: int) -> float:
        M = Bc[j].T @ Bc[i]           # (r, r)
        inner = C[j] @ M @ C[i] @ M.T  # (r, r)
        return float(np.trace(inner)) / (n_rows - 1) ** 2

    D = np.zeros((n, n))
    hsic_diag = [_hsic(i, i) for i in range(n)]
    pairs = list(combinations(range(n), 2))
    for i, j in tqdm(pairs, desc="cka pairs", total=len(pairs)):
        hij = _hsic(i, j)
        denom = np.sqrt(max(hsic_diag[i], 0.0) * max(hsic_diag[j], 0.0))
        cka = float(np.clip(hij / denom, 0.0, 1.0)) if denom > 1e-12 else 0.0
        D[i, j] = D[j, i] = 1.0 - cka
    return list(names), D


# ---------------------------------------------------------------------------
# Bures-Wasserstein distance matrix
# ---------------------------------------------------------------------------

def bures_wasserstein_distance_matrix(
    weights,
    layers: int | list[int] | None = None,
    projections: str | list[str] | None = None,
    align: bool = False,
    cache_dir: Path | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pairwise Bures-Wasserstein distance on the uncentered covariance Σ = P^T P.

    Works entirely in rank-N space (N = n_blocks·r) via thin SVD, where the
    per-block low-rank factors are stacked across every selected (layer,
    proj) block before the SVD:

      B = Q R  (QR of B, per block)
      M = R A  (r×d, per block)         M_total = vstack over blocks  (N×d)
      SVD M_total → U_M, s              [N×d thin SVD]
      M_i M_j^T = M_i,total M_j,total^T  (N×N)
      G = U_Mi^T (M_i M_j^T) U_Mj         (N×N)
      d_BW^2 = ||s_i||^2 + ||s_j||^2 - 2·nuclear_norm(G)

    This is exact, not an approximation: Σ_total = Σ_blocks P^T P =
    M_total^T M_total, and the algorithm above is a function purely of any
    factor M with M^T M = Σ — see docs/notes/frobenius_bw_generalization.md
    for the full proof.

    With ``align=True``, P_i is first rotated by the Procrustes Q* that
    minimises ||P_i Q - P_j||_F before computing the BW distance. This is
    only supported for a single (layer, proj) block (multi-block Procrustes
    alignment is not implemented — raises ``ValueError``). The aligned A_i
    is formed as (A_i @ U_ij) @ V_ij^T — entirely r×r ops.

    Parameters
    ----------
    weights : LoRAWeightCollection
    layers : layer indices to include; default is all layers loaded in *weights*
    projections : projections to include, e.g. ``"o"`` or ``["k", "v"]``;
        accepts short (``"o"``) or long (``"o_proj"``) forms.
        Default is all projections loaded in *weights*.
    align : apply Procrustes alignment to P_i before computing BW
        (single block only)
    cache_dir : directory for caching alignment matrices (None → no cache)
    """
    names = weights.keys()
    n = len(names)

    _layers = _normalize_layers(weights, layers)
    _projs = _normalize_projections(weights, projections)
    blocks = [(layer, proj) for layer in _layers for proj in _projs]

    if align and len(blocks) != 1:
        raise ValueError(
            f"align=True is only supported for a single (layer, projection) "
            f"block; got {len(blocks)} blocks. Multi-block Procrustes "
            f"alignment is not implemented."
        )

    # Per-adapter precompute: stack QR(B)@A across blocks into M_total, then thin SVD.
    Us: list[np.ndarray] = []   # U_M  (N, N) — from unaligned SVD of M_total
    ss: list[np.ndarray] = []   # singular values (N,)
    Ms: list[np.ndarray] = []   # M_total (N, d) — kept for the unaligned cross term
    # Only populated when align=True (exactly one block, per the check above):
    Rs: list[np.ndarray] = []   # R from QR of B  (r, r)
    As: list[np.ndarray] = []   # A  (r, d)
    Bs: list[np.ndarray] = []   # B  (d, r)
    for name in tqdm(names, desc="bw precompute"):
        M_blocks = []
        for layer, proj in blocks:
            A = weights[name].matrix(layer, proj, "A").astype(np.float64)
            B = weights[name].matrix(layer, proj, "B").astype(np.float64)
            _, R = np.linalg.qr(B)                          # R: (r, r)
            M_blocks.append(R @ A)                          # (r, d)
            if align:
                Rs.append(R)
                As.append(A)
                Bs.append(B)
        M_total = np.concatenate(M_blocks, axis=0)          # (N, d)
        U_M, s, _ = np.linalg.svd(M_total, full_matrices=False)
        Us.append(U_M)
        ss.append(s)
        Ms.append(M_total)

    D = np.zeros((n, n))
    pairs = list(combinations(range(n), 2))
    desc = "bw pairs (aligned)" if align else "bw pairs"
    for i, j in tqdm(pairs, desc=desc, total=len(pairs)):
        if align:
            layer, proj = blocks[0]
            U_ij, V_ij, _, _, _ = _load_or_compute_alignment(
                names[i], names[j], As[i], Bs[i], As[j], Bs[j],
                layer, proj, cache_dir,
            )
            # Aligned A_i = A_i Q* = (A_i U_ij) V_ij^T  (all r×r / r×d ops)
            A_i_U = As[i] @ U_ij                      # (r, r): (r,d)@(d,r)
            A_i_aligned = A_i_U @ V_ij.T              # (r, d): (r,r)@(r,d)

            M_i = Rs[i] @ A_i_aligned                 # (r, d)
            U_mi, s_i, _ = np.linalg.svd(M_i, full_matrices=False)

            # M_i M_j^T using aligned A_i
            VjAj = V_ij.T @ As[j].T                   # (r, r): (r,d)@(d,r)
            AiAj = A_i_U @ VjAj                       # (r, r)
            MiMj = Rs[i] @ AiAj @ Rs[j].T             # (r, r)
            G = U_mi.T @ MiMj @ Us[j]                 # (r, r)
            nuclear = float(np.sum(np.linalg.svd(G, compute_uv=False)))
            d2 = float(np.dot(s_i, s_i)) + float(np.dot(ss[j], ss[j])) - 2.0 * nuclear
        else:
            MiMj = Ms[i] @ Ms[j].T                    # (N, N)
            G = Us[i].T @ MiMj @ Us[j]                # (N, N)
            nuclear = float(np.sum(np.linalg.svd(G, compute_uv=False)))
            d2 = float(np.dot(ss[i], ss[i])) + float(np.dot(ss[j], ss[j])) - 2.0 * nuclear
        D[i, j] = D[j, i] = float(np.sqrt(max(d2, 0.0)))
    return list(names), D


# ---------------------------------------------------------------------------
# Cosine similarity matrix
# ---------------------------------------------------------------------------

def cosine_similarity_matrix(
    weights,
    layers: list[int] | None = None,
    projections: str | list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    """Pairwise cosine similarity between concatenated LoRA products B @ A.

    Uses the same low-rank formulation as :func:`frobenius_distance_matrix`
    throughout (no d×d matrices formed), generalized to a concatenation over
    multiple (layer, proj) blocks:

      dot(v_i, v_j) = Σ_{layer,proj} tr(B_i^T B_j · A_j A_i^T)   (all r×r)
      ‖v_i‖^2       = Σ_{layer,proj} _frob_sq(A_i, B_i)

    where v_i is the vector obtained by raveling and concatenating each
    B @ A product across the selected layers/projections.

    Parameters
    ----------
    weights : LoRAWeightCollection
    layers : layer indices to include; default is all layers loaded in *weights*
    projections : projections to include, e.g. ``"o"`` or ``["k", "v"]``;
        accepts short (``"o"``) or long (``"o_proj"``) forms.
        Default is all projections loaded in *weights*.

    Returns
    -------
    names : list of adapter names in matrix order
    S     : (n, n) float64 symmetric cosine similarity matrix, values in [-1, 1]
    """
    names = weights.keys()
    n = len(names)

    _layers = _normalize_layers(weights, layers)
    _projs = _normalize_projections(weights, projections)

    As: list[list[np.ndarray]] = []
    Bs: list[list[np.ndarray]] = []
    norm_sq: list[float] = []
    for name in tqdm(names, desc="cosine precompute"):
        adapter_As: list[np.ndarray] = []
        adapter_Bs: list[np.ndarray] = []
        total = 0.0
        for layer in _layers:
            for proj in _projs:
                A = weights[name].matrix(layer, proj, "A").astype(np.float64)
                B = weights[name].matrix(layer, proj, "B").astype(np.float64)
                adapter_As.append(A)
                adapter_Bs.append(B)
                total += _frob_sq(A, B)
        As.append(adapter_As)
        Bs.append(adapter_Bs)
        norm_sq.append(total)

    S = np.ones((n, n), dtype=np.float64)
    pairs = list(combinations(range(n), 2))
    for i, j in tqdm(pairs, desc="cosine similarity", total=len(pairs)):
        dot = 0.0
        for k in range(len(As[i])):
            K_ij = Bs[i][k].T @ Bs[j][k]                   # (r, r)
            dot += float(np.trace(K_ij @ (As[j][k] @ As[i][k].T)))
        S[i, j] = S[j, i] = dot / np.sqrt(norm_sq[i] * norm_sq[j])

    return names, np.clip(S, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_distance_analysis(
    names: list[str],
    dist_mat: np.ndarray,
    title: str,
    ax_heat,
    ax_mds,
    mds_dim = 2,
    save: bool = False,
    savepath: str | Path | None = None,
) -> None:
    """Draw a heatmap and an MDS scatter for one distance matrix.

    Color conventions match ``notebooks/1_plots_vis_workshop.ipynb``:
    - Heatmap: ``copper_r``
    - Scatter:  ``plasma``, colored by topic-0 percentage

    Parameters
    ----------
    names:
        Adapter names in matrix order.
    dist_mat:
        (n, n) symmetric distance matrix.
    title:
        Subplot title.
    ax_heat, ax_mds:
        Matplotlib axes for the heatmap and MDS scatter respectively.
    mds_dim:
        Number of dimensions for the MDS embedding.
    save:
        Save the figure that owns *ax_heat* / *ax_mds*. Implied by *savepath*.
    savepath:
        Where to write it; defaults to ``figures/fig_<title>.png`` under the
        repo base. A bare filename lands in that same ``figures/`` directory,
        while a path with directories (or an absolute path) is used as given.
        A non-image suffix gets ``.png`` appended.

        When several calls share one figure — the usual grid layout, one call
        per column — only the last call writes a fully populated image.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.manifold import MDS

    from src.plots import save_figure

    labels = [model_label(n) for n in names]
    values = [float(l) for l in labels]

    # heatmap
    sns.heatmap(
        dist_mat,
        ax=ax_heat,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="copper_r",
        square=True,
    )
    ax_heat.set_title(title)

    # MDS scatter
    scatter_cmap = plt.cm.plasma
    scatter_norm = mcolors.Normalize(vmin=0, vmax=100)

    coords = MDS(n_components=mds_dim, dissimilarity="precomputed", random_state=42).fit_transform(dist_mat)

    if mds_dim == 1:
        ax_mds.set_xlabel("MDS Dimension 1")
        coords = np.hstack([coords, np.zeros_like(coords)])
        ax_mds.set_yticks([])
    else:
        ax_mds.set_xlabel("MDS Dimension 1")
        ax_mds.set_ylabel("MDS Dimension 2")

    for (x, y), label, val in zip(coords, labels, values):
        color = scatter_cmap(scatter_norm(val))
        ax_mds.scatter(x, y, color=color, s=100, zorder=3)
        ax_mds.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5),
                        fontsize=plt.rcParams.get("font.size", 8))

    sm = plt.cm.ScalarMappable(cmap=scatter_cmap, norm=scatter_norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_mds, shrink=0.8)
    cb.set_label("topic 0 (%)")
    tick_vals = sorted(set(values))
    cb.set_ticks(tick_vals)
    cb.set_ticklabels([str(int(v)) for v in tick_vals])

    ax_mds.axis("equal")
    ax_mds.set_title(title + " (MDS)")

    if save or savepath is not None:
        save_figure(ax_heat.get_figure(), savepath, title)
