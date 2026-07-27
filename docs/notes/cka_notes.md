# CKA: current status and open design question for multi-block sweeps

## Status

`cka_distance_matrix` in `src/notebook/structure.py` has **not** been
generalized to accept lists of layers/projections. It still only takes a
single `layer: int` and single `proj: str`, unlike `cosine_similarity_matrix`,
`frobenius_distance_matrix`, and `bures_wasserstein_distance_matrix`, which
now all accept `layers`/`projections` as lists and sum contributions across
`(layer, proj)` blocks (see `frobenius_bw_generalization.md`).

This wasn't an oversight — a first pass derived a mathematically valid
generalization for CKA, but it depends on a genuinely non-obvious modeling
choice, and rather than pick that choice unilaterally, it's parked here for
discussion.

## What HSIC/CKA mean

**HSIC** = Hilbert-Schmidt Independence Criterion, a kernel-based measure of
statistical dependence between two sets of paired observations, computed
from their Gram/kernel matrices. **CKA** (Centered Kernel Alignment)
normalizes HSIC to `[0, 1]` by the HSIC of each kernel with itself, making
it comparable across different scales.

## Why CKA doesn't generalize the same way as Frobenius/BW

Frobenius and BW both reduce to sums/stacks over a shared axis:
`d_in` (3072) is constant across every layer and projection, so blocks can
always be concatenated along the column space they already share, with no
ambiguity about what's being combined.

CKA is different: its kernel `K = P P^T` is built over the **sample axis**
— the `d_out` rows of `P` — i.e. "samples" = *output neurons within one
`(layer, proj)` block*. There is no axis shared across different blocks the
way `d_in` is; each block's `d_out` rows are a disjoint, differently-sized
set of neurons (GQA even makes `d_out` differ between `k`/`v` and `q`/`o`).
So "combine multiple blocks" requires deciding what a pooled sample space
*means*, and that choice changes what question CKA is answering.

### Option 1 (worked out, not implemented): pool neurons across blocks

Stack `P^{(k)}` blocks along the sample axis (valid since columns/`d_in`
match across blocks) into one pooled sample space of size `Σ d_out`, and run
linear CKA over that pooled set — "samples" = neurons pooled from every
selected layer/projection, "features" = `d_in`.

The low-rank form (derived, not implemented):
$$
T_{ij} = P_{i,\text{total},c}^T P_{j,\text{total},c} = \sum_{k} (A_i^{(k)})^T M_{ij}^{(k)} A_j^{(k)}, \quad M_{ij}^{(k)} = (Bc_i^{(k)})^T Bc_j^{(k)}
$$
which is **not** a simple sum of per-block HSIC values (that would drop the
cross terms in $\|T_{ij}\|_F^2$). The correct closed form is
$T_{ij} = A_{i,\text{total}}^T \cdot M_{bd} \cdot A_{j,\text{total}}$, where
$A_{*,\text{total}} = \mathrm{vstack}_k(A_*^{(k)})$ (`N×d_in`) and $M_{bd}$
is the `N×N` **block-diagonal** matrix with each block's `r×r` $M_{ij}^{(k)}$
on the diagonal (off-diagonal blocks are exactly zero — different blocks'
centered `B`s never interact, since they came from disjoint sample sets).
Then, mirroring the existing single-block formula
(`inner = C_j @ M @ C_i @ M.T`):
$$
\mathrm{HSIC}(K_i,K_j) = \mathrm{tr}(T_{ij}^TT_{ij}) = \mathrm{tr}\big(C_{j,\text{total}}\,M_{bd}\,C_{i,\text{total}}\,M_{bd}^T\big) \big/ (\textstyle\sum d_{\text{out}} - 1)^2
$$
where $C_{*,\text{total}} = A_{*,\text{total}}A_{*,\text{total}}^T$
(`N×N`, dense). This reduces exactly to today's single-block formula when
there's one block (block-diagonal collapses to the single dense block), and
avoids ever forming a `d×d` matrix — but building `M_bd` explicitly per pair
costs `O(N²)` memory and `O(N³)` compute (`N` up to ~1792 for the largest
sweep config), noticeably more expensive than the Frobenius/BW
generalizations at the same block count.

### Option 2 (raised by the user, not worked out): each block is a sample

A materially different framing: treat each `(layer, proj)` **block itself**
as one "sample", with the block's (some representation of its) vectorized
weights as the feature vector — i.e. "samples" = *which LoRA block*,
"features" = *that block's weights*. This is closer in spirit to how
`cosine_similarity_matrix` already treats the whole concatenated vector as
one point per adapter, rather than pooling individual output neurons within
blocks. It would answer a different question than Option 1 (something like
"do these two adapters' layers/projections vary in a similar pattern across
the sweep?" rather than "are the neuron-level representations aligned?").
Not derived yet — would need its own low-rank formulation before
implementing, and it's not obvious what the right per-block feature
representation would be (the raw `B@A` product is `d_out×d_in` and varies
in shape across blocks under GQA, so it isn't immediately a fixed-length
feature vector without some further reduction).

## Open question for next discussion

Which (if either) of these framings is the right generalization of CKA for
this project's purposes — or is there a third option? Worth deciding before
implementing, since Options 1 and 2 measure genuinely different things, not
two approximations of the same quantity.
