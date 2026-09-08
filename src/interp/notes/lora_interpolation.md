# Interpolating LoRA adapters between simplex corners

`src/interp/` asks two questions about a family of adapters trained on mixtures
of `k` dataset components. Writing `ΔW_T` for the effective update (multiplied BA-style) of an adapter
trained on mixture `w`, and `ΔW_1 … ΔW_k` for the adapters trained on the pure
components:

1. **Weight similarity** — how close is `Σ wₖ ΔWₖ` to `ΔW_T` for prefixed coefficients `w`, pulled from the data mixture ?
2. **Coefficient recovery** — which coefficients actually fit best if reconstructing `ΔW_T` from pure components, and how far are they from true `w`?

## What a "block" is

This note calls each BA-style adapter module a **block**, keyed by
the `(layer, module)` pair it sits at: i.e. `(0, in_proj_qkv)`, `(7, o_proj)`, and so
on. `src/interp/loading.py::load_adapter` returns exactly that dict:

```python
{(layer, module): {"A": (r, d_in), "B": (d_out, r)}, ...}
```

For the Qwen3.5-4B adapters here there are **104 blocks**, because the model is
a hybrid stack — 24 gated-delta `linear_attn` layers with 3 adapted modules
each, and 8 full-attention `self_attn` layers with 4 each (24·3 + 8·4 = 104).
Every layer index 0–31 belongs to exactly one family, so `(layer, module)` is
unique; `load_adapter` asserts this rather than assuming it.

| module | family | layers | `A` | `B` | `ΔW` per layer | params |
|---|---|---|---|---|---|---|
| `in_proj_qkv` | `linear_attn` | 24 | (16, 2560) | (8192, 16) | (8192, 2560) | 503.3 M |
| `in_proj_z`   | `linear_attn` | 24 | (16, 2560) | (4096, 16) | (4096, 2560) | 251.7 M |
| `out_proj`    | `linear_attn` | 24 | (16, 4096) | (2560, 16) | (2560, 4096) | 251.7 M |
| `q_proj`      | `self_attn`   |  8 | (16, 2560) | (8192, 16) | (8192, 2560) | 167.8 M |
| `k_proj`      | `self_attn`   |  8 | (16, 2560) | (1024, 16) | (1024, 2560) |  21.0 M |
| `v_proj`      | `self_attn`   |  8 | (16, 2560) | (1024, 16) | (1024, 2560) |  21.0 M |
| `o_proj`      | `self_attn`   |  8 | (16, 4096) | (2560, 16) | (2560, 4096) |  83.9 M |

Note `r = 16` everywhere while `d_in` and `d_out` vary by module. That is what
makes the rank-space trick below pay: the small dimension is shared.

## What `ΔW_i` is

Per module, PEFT adds `ΔW = (α/r)·B A` at inference. Writing `s = α/r` (here
`32/16 = 2`) and `b` for a block:

$$\Delta W_i^{(b)} = s\, B_i^{(b)} A_i^{(b)} \qquad \text{shape } (d_{\text{out}}^{(b)},\, d_{\text{in}}^{(b)})$$

`ΔW_i` is the whole collection of 104 such
matrices, i.e. everything adapter `i` adds to the base model. To compare two
adapters, treat that collection as **one long vector** `v_i`: ravel each block
and concatenate them in a fixed block order.

$$v_i = \big(\operatorname{vec}(\Delta W_i^{(1)}),\; \ldots,\; \operatorname{vec}(\Delta W_i^{(104)})\big) \in \mathbb{R}^{1.3\times 10^9}$$

Summing the `params` column gives its length: **1,300,234,240 entries** — 5.2 GB at float32,
10.4 GB at the float64 this code computes in, per adapter, and there are 16 of
them. It is never built.

Because `vec` is an isometry and concatenation partitions the vector's indices,
the inner product decomposes blockwise with no approximation:

$$\langle v_i, v_j\rangle = \sum_b \langle \Delta W_i^{(b)}, \Delta W_j^{(b)}\rangle_F = s^2 \sum_b \langle B_i^{(b)}A_i^{(b)},\; B_j^{(b)}A_j^{(b)}\rangle_F$$

(the same decomposition argument as `frobenius_bw_generalization.md`, Claim A).

## The Gram matrix

These are the different mixtures: `G` is the `n × n` matrix of inner products over the `n = 16` adapters.

$$G_{ij} = \langle v_i, v_j \rangle$$

`G` is symmetric and PSD, `G_ii = ‖v_i‖²`, and it is *all* the information this
analysis needs. This is because we can rewrite the squared error between `v_T` and
the mixture of corners in terms of the gram entries.
With `G_TT = ⟨v_T, v_T⟩`, `g = (⟨v_T, v_k⟩)_k` for the `k`
corners, and `Gc = (⟨v_i, v_j⟩)_{i,j ∈ corners}`:

$$\Big\|v_T - \sum_k x_k v_k\Big\|^2 = G_{TT} - 2\,x^\top g + x^\top G_c\, x$$

— a quadratic in `x`, minimized at `x^* = G_c^{-1} g`. Residuals, cosines,
distances and fitted coefficients all follow from a `3×3` and a `3`-vector,
without touching a weight tensor again.

`LoRAGram` stores the **per-block** contributions, shape `(104, 16, 16)`, and
sums over the block axis on demand (`.total`). That is 213 KB and it is what
makes restricting to a subset of layers or modules — and the per-block error
profile — free after the fact.

## Computing `G` without building `ΔW`

By the cyclic property of the trace, one block's contribution needs only `r×r`
matrices:

$$\langle B_iA_i,\; B_jA_j\rangle_F = \operatorname{tr}\!\big(A_i^\top B_i^\top B_j A_j\big) = \operatorname{tr}\!\big((B_i^\top B_j)(A_j A_i^\top)\big) = \sum_{a,b} (B_i^\top B_j)_{ab}\,(A_i A_j^\top)_{ab}$$

using `tr(XY) = Σ X ⊙ Yᵀ` at the last step. `B_iᵀB_j` is `(r, r)` and `A_iA_jᵀ`
is `(r, r)`, so the cost per pair is `O(r²(d_out + d_in))` instead of
`O(d_out·d_in)` — for `in_proj_qkv`, about 2.8 M flops rather than 21 M, and no
21-million-entry intermediate. This is the identity
`src/notebook/structure.py` already uses for its pairwise builders.

`compute_gram` goes one step further and gets **all `n²` pairs of a block from
two GEMMs** rather than `n²` small ones. For one block, with `n = 16` adapters
and `r = 16`:

```python
B_all = np.concatenate([W[nm][blk]["B"].T for nm in names], axis=0)  # (n*r, d_out) = (256, 8192)
A_all = np.concatenate([W[nm][blk]["A"]   for nm in names], axis=0)  # (n*r, d_in)  = (256, 2560)

BB = (B_all @ B_all.T).reshape(n, r, n, r)   # (256,256) -> (16,16,16,16)
AA = (A_all @ A_all.T).reshape(n, r, n, r)

per_block[bi] = np.einsum("iajb,iajb->ij", BB, AA)   # (16, 16)
```

Reading the indices: `BB[i, a, j, b]` is row `a` of `B_iᵀ` dotted with row `b`
of `B_jᵀ`, which is exactly `(B_iᵀB_j)[a, b]`; likewise `AA[i, a, j, b] =
(A_iA_jᵀ)[a, b]`. The einsum contracts over `a` and `b` while keeping `i` and
`j` free, so entry `(i, j)` is `Σ_ab (B_iᵀB_j)_{ab} (A_iA_jᵀ)_{ab}` — the
formula above, for every pair at once. Finally `per_block *= s²` applies the
`α/r` scale once, globally.

The `reshape` assumes every adapter has the same rank *in that block*, which is
checked per block; a mixed-rank block falls back to an explicit pairwise loop.
That is not hypothetical — a combination of `k` rank-`r` adapters has rank up to
`k·r`, which is exactly how `check_interpolation.py` builds its exact-recovery
target. Both paths are checked against the materialized dense product and agree
to ~1e-16.

**Threading matters more than the algorithm here.** The GEMMs are small enough
that thread dispatch dominates: on this cluster (BLAS reporting 128 threads,
OpenMP 256) the full 16-adapter, 104-block Gram takes **77 s unpinned and 4.0 s
under `threadpool_limits(1)`**. `compute_gram` pins it itself rather than
trusting the caller to.

## Two conventions that differ from the rest of the repo

**The `α/r` scale is applied here and nowhere else.**
`src/notebook/structure.py` and `src/metrics/frobenius.py` operate on the
unscaled `B @ A`; `src/taxonomy/structural.py` only records `lora_alpha` in
metadata. Since `α/r` is shared by every adapter in a collection, it cancels
from cosines, normalized coefficients and relative errors, and multiplies
squared distances by exactly `s²` (4.0 for `α=32, r=16`). **This discrepancy is
intended.** Do not "reconcile" the two: retrofitting the scale into the existing
builders would silently move every stored structural distance and every number
in the measurement tables under `docs/notes/`.

**Storage is float64, not float32.** `DistanceMatrix.save` and
`SimplexProjection.save` both downcast. Here the Gram is not an end product — a
`k×k` submatrix of it is solved against to produce every coefficient — so the
digits are load-bearing. The whole per-block tensor is 213 KB.

## Why the primary estimator is the *normalized unconstrained* fit

Three estimators are reported: the true coefficients `w`, the unconstrained
minimizer `x^* = G_c^{-1}g`, and the simplex-constrained minimizer
(`w ≥ 0`, `Σw = 1`, solved exactly by enumerating the `2^k − 1` faces — for
`k=3` that is 7 tiny KKT solves, cheaper and more accurate than an iterative QP).

The unconstrained fit consistently sums to **γ ≈ 0.73, not 1**. Forcing `Σw = 1`
fights that shrinkage and pushes the surplus onto the components that should be
zero, roughly doubling the coefficient error. So `γ = Σx^*` is reported as its
own scalar — it is a real property of the geometry, not an artifact — and the
*direction* is read off `x^*/γ`.

| estimator | mean ‖ŵ − w‖₂ |
|---|---|
| normalized unconstrained `x*/γ` | **0.053** |
| simplex-constrained | 0.117 |
| uniformly random simplex point | 0.533 |

## Relative error

Residuals are reported relative to the target's own norm, so mixtures of
different magnitudes are comparable:

$$\mathrm{rel}(x) = \frac{\big\|v_T - \sum_k x_k v_k\big\|}{\|v_T\|} = \sqrt{\frac{G_{TT} - 2\,x^\top g + x^\top G_c\,x}{G_{TT}}}$$

0 is an exact reconstruction; 1 is what predicting nothing (`x = 0`) scores.
`fitting.py::residual_sq` evaluates the numerator straight from the Gram, with a
clip at zero that only absorbs float error when the residual is genuinely ~0 (a
corner fitting itself). Three coefficient vectors go in, giving the three
columns of the results table:

| reported as | `x` |
|---|---|
| `rel_true` | the true data proportions `w` |
| `rel_ols` | the unconstrained minimizer `x^* = G_c^{-1}g` |
| `rel_simplex` | the constrained minimizer (`x ≥ 0`, `Σx = 1`) |

`rel_ols` is the minimum over all `x`, so it is the best *any* blend of corners
can do. At that minimizer the residual is orthogonal to the corner span, so
`1 − rel_ols²` is the fraction of `‖v_T‖²` the span captures — 30% here, and the
form the generated `report.md` quotes. Numerator and denominator both carry
`s = α/r`, so the ratio is free of it.

The same expression over one block's contribution instead of `.total` gives the
per-block profile (`plots.py::per_block_rel_error`) with no weights reloaded.

## What the measurements say

On the 16 Qwen3.5-4B adapters over the three Yahoo topic groups (13 mixtures,
3 corners, 104 modules):

| quantity | value |
|---|---|
| relative error at the **true (data)** coefficients | 0.862 (0.76–0.93) |
| relative error at the **best-fit (unnormalized)** coefficients | 0.836 |
| cosine to the true-coefficient interpolation | 0.43–0.67 |
| coefficient error ‖ŵ − w‖₂ | 0.053 |
| shrinkage γ | 0.727 ± 0.044 |
| corner–corner cosine | 0.069–0.073 |

Three readings, all simultaneously true:

- **The corner span is the wrong subspace.** Relative error ≈ 0.84 even at the
  best coefficients: most of the trained adapter's magnitude lies outside the
  3-dimensional corner span. The corners are nearly mutually orthogonal
  (cosine ≈ 0.07), so independently trained LoRAs land in largely disjoint
  directions and span very little of the ambient space. Interpolating corner
  adapters does **not** reconstruct the mixture adapter's weights.
- **Within that subspace the direction is nearly exact.** Coefficients recover
  the data proportions 10× better than chance.
- **The magnitude is systematically over-predicted**, by a strikingly consistent
  factor: γ = 0.727 ± 0.044 across every mixture.

The per-block breakdown rules out an architecture artifact: mean relative error
is 0.833 over the 72 `linear_attn` modules and 0.844 over the 32 `self_attn`
modules.


## Does the *configuration* recover the simplex?

`coeff_l2` scores one adapter at a time. It cannot answer whether the fitted
coefficients, taken together, lay out the same arrangement the data recipes do —
a set of fits could each be a little off yet reproduce the shape exactly, or each
be close while scrambling which mixture neighbours which. That second question is
the one `src/analysis/comparison.py` asks of every other taxonomy level, so
asking it here is what puts coefficient recovery on a comparable scale.

`src/interp/geometry.py` builds a distance matrix over the fitted coefficients,
embeds it, and scores it against the ideal simplex with the two statistics the
rest of the repo uses. Over all 16 adapters:

| coefficients | Procrustes disparity ↓ | dCor ↑ |
|---|---|---|
| `ols_normalized` | **0.00637** | **0.9902** |
| `simplex` | 0.03262 | 0.9841 |

Excluding the three corners, which fit themselves exactly and so pin three of
sixteen points to the vertices: 0.01012 vs 0.01769, dCor 0.9858 vs 0.9808.

The ordering is the same one `coeff_l2` reports per adapter (0.053 vs 0.117), and
for the same reason: forcing `Σw = 1` fights the shrinkage γ instead of reporting
it, and pushes the surplus onto components that should be zero. The defect that
doubles the per-point error also deforms the arrangement. Two measurements, one
finding.

The coefficient distance matrix is built by the same function that builds the
truth's — `simplex_distance_matrix`, which reads each coefficient vector as
barycentric coordinates and places it on a unit-edge regular simplex. For rows
summing to 1 that is an isometry up to `1/√2`, and both statistics are
scale-invariant, so **it changes no reported number**; taking `pdist(W)` directly
agrees to eight decimals. It is done anyway so both sides of the comparison come
from one function and cannot drift apart in units, and so the redundant `k`th
dimension is dropped before the embedding. The isometry holds *only* because the
rows sum to 1 — the raw `ols` sums to γ, which the embedding projects away, which
is why it is not one of the scored estimates.

Two things the numbers do not say:

- **The MDS step is doing no work at k=3.** The coefficient distance matrix is
  exactly Euclidean in `k−1 = 2` dimensions, so embedding it at
  `n_components = 2` reproduces the arrangement rather than approximating it:
  stress ≈ 1e-5, and the disparity computed straight from the coefficients
  matches the one through MDS to four decimals. Both are reported. The step
  earns its place at 4+ components or a lower projection dimension, and in
  making these numbers commensurate with the other levels — not here.
- **dCor is unsigned and saturated.** It scores 1.0 for an arrangement that
  reproduces the truth exactly *backwards*, and at these sizes it separates the
  two estimates by under 0.01. Rank by Procrustes; read dCor as confirmation
  that the structure is present at all.

Run it standalone against a finished run — no adapters, no Gram, a few seconds:

```
python scripts/run_coefficient_geometry.py --run-dir results/lora_interpolation
```

It needs a scikit-learn new enough for `MDSGeometry`'s `metric_mds=` keyword
(the project `taxonomy` env has it); `--geometry pca` is classical MDS in pure
numpy and has no such requirement.

## Not covered here

Whether the interpolated adapter *behaves* like the trained one. Relative error
≈ 0.84 alongside coefficient error ≈ 0.05 makes that the natural follow-up, and
it needs the functional/behavioral taxonomies and GPU jobs. The fitted
coefficients are written to `fits/fits.safetensors` so that work can start
without recomputation.

The other open question is whether this geometry survives at more components.
Everything above is `k = 3`, where the simplex is a triangle and the embedding
is exact; `k ≥ 4` is where the MDS step starts doing real work and where the
projection dimension becomes a choice rather than a formality.
