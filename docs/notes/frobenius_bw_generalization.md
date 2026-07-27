# Generalizing Frobenius and Bures-Wasserstein distances to multi-layer, multi-projection sweeps

`src/notebook/structure.py` computes pairwise distances between LoRA
weight-delta products `P = B @ A` (`B`: `d_out×r`, `A`: `r×d_in`, `r=16`)
without ever materializing the full `d×d` product. Originally this only
worked for a single `(layer, proj)` block at a time. `frobenius_distance_matrix`
and `bures_wasserstein_distance_matrix` now also accept **lists** of layers
and projections (`layers: int | list[int] | None`,
`projections: str | list[str] | None`), matching
`cosine_similarity_matrix`. This note records the proofs that justify why
that generalization is exact, not an approximation.

Both proofs lean on one shared fact: `d_in` (3072) is the same for every
projection (`k`, `q`, `v`, `o`) and every layer — only `d_out` varies (GQA
gives `k_proj`/`v_proj` a smaller `d_out`). So `A^{(k)}` (`r × d_in`) always
shares its column space across blocks, which is what makes vertical
stacking across blocks valid below.

## Frobenius

**Claim A (norm preservation under concatenation).** Let
$v_i = \big(\mathrm{vec}(P_i^{(1)}), \ldots, \mathrm{vec}(P_i^{(K)})\big)$ be the
concatenation of $K$ raveled blocks $P_i^{(k)} = B_i^{(k)}A_i^{(k)}$. Since
$\mathrm{vec}$ is by definition an isometry onto the Frobenius norm,
$\|\mathrm{vec}(X)\|_2^2 = \sum_{a,b} X_{ab}^2 = \|X\|_F^2$ for *any* matrix
$X$. Squared Euclidean distance decomposes over any partition of vector
indices — concatenation is exactly such a partition:
$$
\|v_i-v_j\|_2^2 = \sum_{\text{idx}} (v_i-v_j)_{\text{idx}}^2
= \sum_{k=1}^K \sum_{\text{idx}\in \text{block }k} (v_i-v_j)_{\text{idx}}^2
= \sum_{k=1}^K \big\|\mathrm{vec}(P_i^{(k)}) - \mathrm{vec}(P_j^{(k)})\big\|_2^2
= \sum_{k=1}^K \big\|P_i^{(k)}-P_j^{(k)}\big\|_F^2 .
$$
So the squared Euclidean distance between the concatenated vectors equals
the sum of per-block squared Frobenius distances — an identity, not an
approximation. Expanding each block via
$\|X-Y\|_F^2 = \|X\|_F^2+\|Y\|_F^2-2\langle X,Y\rangle_F$ and
$\langle P_i^{(k)},P_j^{(k)}\rangle_F = \mathrm{tr}\big((B_i^{(k)T}B_j^{(k)})(A_j^{(k)}A_i^{(k)T})\big)$
(all `r×r` ops), and summing over $k$, gives exactly what the code computes:
`trs_i + trs_j - 2·Σcross`, summed over blocks instead of evaluated for one.

**Claim B (relationship to cosine similarity — law of cosines).** With
$a=\|v_i\|,\; b=\|v_j\|,\;\cos_{ij}=\dfrac{\langle v_i,v_j\rangle}{ab}$:
$$
d_{\text{frob}}(v_i,v_j)^2 = \|v_i-v_j\|_2^2 = \|v_i\|^2+\|v_j\|^2-2\langle v_i,v_j\rangle
= a^2+b^2-2ab\cos_{ij}.
$$
This is exactly the **law of cosines**: $v_i,v_j$ and the origin form a
triangle with sides $a,b,d_{\text{frob}}$ and included angle
$\theta=\arccos(\cos_{ij})$. It's not a special property of LoRA weights —
it holds for any two vectors in Euclidean space — but it does mean Frobenius
distance and cosine similarity, computed on the *same* concatenated
vectors, are never independent: they're two views (radial vs. angular) of
the same triangle. Only when $a=b$ (equal norms) does it collapse to a pure
function of $\cos_{ij}$:
$$
a=b \;\Rightarrow\; d_{\text{frob}}^2 = 2a^2(1-\cos_{ij}) \;\Rightarrow\; d_{\text{frob}} = a\sqrt{2(1-\cos_{ij})},
$$
a monotonic (nonlinear, square-root) rescaling of $1-\cos_{ij}$. Whether
this holds tightly in practice depends on how close `‖v_i‖` values are
across adapters for a given config — not assumed, worth checking
empirically per config if the relationship matters for an analysis.

## Bures-Wasserstein

**Setup.** For a single block, $B^{(k)}=Q^{(k)}R^{(k)}$ (QR), and since
$Q^{(k)T}Q^{(k)}=I$: $B^{(k)T}B^{(k)} = R^{(k)T}R^{(k)}$. With
$M^{(k)} := R^{(k)}A^{(k)}$ ($r\times d_{\text{in}}$):
$$
\Sigma_i^{(k)} := P_i^{(k)T}P_i^{(k)} = A_i^{(k)T}B_i^{(k)T}B_i^{(k)}A_i^{(k)}
= A_i^{(k)T}R_i^{(k)T}R_i^{(k)}A_i^{(k)} = M_i^{(k)T}M_i^{(k)}.
$$

**Claim (stacking).** Define $M_{i,\text{total}} := \mathrm{vstack}_k\big(M_i^{(k)}\big)$
($N\times d_{\text{in}}$, $N=Kr$ — valid since every block shares the same
$d_{\text{in}}=3072$ column space). Using the identity
$(\mathrm{vstack}_k X_k)^T(\mathrm{vstack}_k Y_k) = \sum_k X_k^TY_k$ (a row
partition, same mechanism as Claim A, applied here to an outer product
instead of a scalar):
$$
M_{i,\text{total}}^T M_{i,\text{total}} = \sum_{k=1}^K M_i^{(k)T}M_i^{(k)} = \sum_{k=1}^K \Sigma_i^{(k)} =: \Sigma_{i,\text{total}}.
$$
So $M_{i,\text{total}}$ is a valid factor of the *pooled* covariance
$\Sigma_{i,\text{total}} = \sum_k P_i^{(k)T}P_i^{(k)}$ (itself PSD, being a
sum of PSD matrices).

**Claim (factor-agnosticism of $d_{BW}$).** Bures-Wasserstein distance is,
by its textbook definition, a function purely of the two PSD matrices:
$$
d_{BW}(\Sigma_i,\Sigma_j)^2 = \mathrm{tr}(\Sigma_i)+\mathrm{tr}(\Sigma_j) - 2\,\mathrm{tr}\Big(\big(\Sigma_i^{1/2}\Sigma_j\Sigma_i^{1/2}\big)^{1/2}\Big).
$$
It does not reference *how* $\Sigma$ was obtained or factored — any $M$ with
$M^TM=\Sigma$ is interchangeable for computing it. The existing code
([structure.py](../../src/notebook/structure.py)) is precisely an algorithm
that takes such an $M$ and computes $d_{BW}$ from it via QR/thin-SVD/nuclear
norm — every quantity it uses (`U_M`, `s` from `svd(M)`, and `M_i@M_j.T` for
the cross term) is expressible purely in terms of $M$, never referencing
$A,B,R$ individually beyond having built $M$ from them. Therefore, since
$M_{i,\text{total}}^TM_{i,\text{total}}=\Sigma_{i,\text{total}}$ exactly,
running the *identical, unmodified algorithm* on $M_{i,\text{total}}$ (in
place of the single-block $M_i$) must return
$d_{BW}(\Sigma_{i,\text{total}},\Sigma_{j,\text{total}})$ — the correct
pooled-block Bures-Wasserstein distance. It's the same algorithm applied to
a bigger (but still $\ll d\times d$) input, justified because the
algorithm's correctness only ever depended on the algebraic relationship
$M^TM=\Sigma$, which is preserved by stacking.

**Consistency check** (verified numerically when this was implemented):
$$
\|s_{i,\text{total}}\|^2 = \mathrm{tr}(\Sigma_{i,\text{total}}) = \sum_k \mathrm{tr}(\Sigma_i^{(k)}) = \sum_k\|P_i^{(k)}\|_F^2 = \sum_k \mathrm{tr}_i^{(k)}
$$
— the sum of squared singular values of $M_{i,\text{total}}$ equals the
same per-block trace sum used in the Frobenius generalization.

## Practical notes

- Both functions accept `align=True` only for a single `(layer, proj)` block
  (raises `ValueError` otherwise) — Procrustes alignment has not been
  generalized to multi-block sweeps; the existing single-block aligned code
  path is untouched and still used by `notebooks/3_lora_weight_investigation.ipynb`.
- Sizing: the largest sweep config used so far is `k+q+v+o` × all 28 layers
  = 112 blocks, `N = 112×16 = 1792`. The dominant cost is the `N×d_in` thin
  SVD in Bures-Wasserstein (`M_total` is `1792×3072`) — this is the slow
  step, not the Frobenius side, which stays cheap (`r×r` sums) regardless of
  block count.
- `cka_distance_matrix` was **not** generalized alongside these two — see
  `cka_notes.md` for why.
