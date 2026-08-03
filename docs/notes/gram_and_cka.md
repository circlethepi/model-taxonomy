# Gram matrices and CKA in this repo

Written because "how is CKA implemented here?" has **two answers**, and because
the word `gram` names three different objects across the taxonomy levels. Both
are easy to get wrong in a way that produces a plausible number rather than an
error.

Companion note: [`cka_notes.md`](cka_notes.md) covers the open design question
for the *structural* implementation described in §1.2. This note covers what the
two implementations are, what their inputs mean, and where the traps are.

---

## 1. There are two CKA implementations, and they are not comparable

### 1.1 `src/metrics/cka.py::CKADistanceMetric` — the pipeline path

Takes two `ModelRepresentation`s and forms its own kernel from their **rows**:

```python
K = X @ X.T      # X is rep.matrix, shape (n_rows, d)
L = Y @ Y.T
distance = 1 - CKA(K, L)
```

This is what `build_taxonomy_artifacts(..., metric="cka")` uses for every level
except structural. Linear kernel by default; RBF available with a median-heuristic
bandwidth.

### 1.2 `src/notebook/structure.py::cka_distance_matrix` — the structural path

A LoRA-specific low-rank form that never materialises `B @ A`. Its samples are
**output neurons within one `(layer, projection)` block** — the `d_out` rows of
the product — with `d_in` as the feature axis.

### The difference that matters

Same name, same normalization, **different sample axis**, therefore a different
question:

| | `src/metrics/cka.py` | `src/notebook/structure.py` |
|---|---|---|
| a "sample" is | a row of `rep.matrix` | one output neuron of one LoRA block |
| a "feature" is | that row's coordinates | `d_in` |
| asks | are these models' *response geometries* aligned? | are these adapters' *neuron-level weight* representations aligned? |

Two numbers from these are not on the same scale and should not be tabulated in
one column without saying which produced them.

---

## 2. What a row means at each level

CKA's value is entirely determined by what a row is, and this is the thing most
worth knowing before reading any distance:

| level | a row of `rep.matrix` is | shape |
|---|---|---|
| `functional` | **a query** | `(n_queries, L·d)` in the default `concat` view |
| `behavioral` | **a query** (its generated text, embedded) | `(n_queries, d)` |
| `dataset_embedding` | **a document** — or a single pooled row in the `mean` / `gram` representations | `(N, d)` / `(1, ·)` |
| `structural` | **a neuron or block** | see §1.2 |

`functional` and `behavioral` share a row semantics *and*, when run from the
paired smoke configs, the same query draw — which is what makes those two levels
directly comparable rather than merely thematically related.

---

## 3. `gram` names three different objects

Be specific about which one is meant:

| where | definition | shape | rows are |
|---|---|---|---|
| `functional`, current | `G = H Hᵀ` of the concatenated feature matrix | `(n_queries, n_queries)` | queries |
| `functional`, **removed** | stacked upper triangles of *per-layer* Gram matrices | `(n_layers, n_queries(n_queries+1)/2)` | **layers** |
| `dataset_embedding` | flattened upper triangle of `E Eᵀ` | `(1, N(N+1)/2)` | one pooled row |

The middle row is the historical form. It was dropped rather than kept as an
option: two things called "gram" with different row semantics is exactly the
confusion this note exists to prevent. Nothing had been computed with it —
functional had never been run — so there is nothing to migrate.

The third row is a *vector*, not a matrix. It shares a name with the first and
almost nothing else.

---

## 4. The double-kernel trap

**A stored Gram is a kernel, not a feature matrix.**

`CKADistanceMetric.compute` forms `K = X Xᵀ` from whatever it is handed. Hand it
a Gram `G = H Hᵀ` and it computes

```
G Gᵀ = (H Hᵀ)²
```

which is a different quantity that still returns a finite number in `[0, 1]`.
Nothing about the output looks wrong.

**The rule:** feed CKA the `concat` view. Linear CKA already forms exactly this
Gram internally, so nothing is lost by not pre-computing it. The `gram` view is
for inspection and for kernel-aware consumers.

This is enforced rather than documented: `ActivationCache.load` tags kernel views
with `metadata["is_kernel"]`, and `CKADistanceMetric.compute` raises on them.
`scripts/check_analysis.py::t_cka_row_guard` pins the refusal.

---

## 5. The unbiased estimator has a domain

`_hsic(..., unbiased=True)` — the default — uses the Song et al. (2012)
estimator, whose denominator is `n(n-3)`:

```
rows 3 -> nan   (division by zero)
rows 4 -> 1.0   (degenerate)
rows 8 -> 0.985
```

Measured, not derived. At `n = 3` this returned NaN, which then flowed into a
distance matrix and an MDS fit, where the cause is far harder to locate than at
the point it occurred. `CKADistanceMetric` now raises below 4 rows and names
`unbiased=False` as the escape hatch, which genuinely works at small `n`.

This mattered because the *old* `gram` definition made rows = layers:
`experiments/yahoo_topics.yaml` pairs `functional: cka` with four
`layer_indices`, i.e. a 4-row matrix — one row away from NaN. Under the current
definition rows are queries (64 in the smoke config), so the default path cannot
reach the degenerate regime. The guard stays because nothing stops a caller from
comparing four of something.

---

## 6. What the metric does not check

`compute` verifies that `a.n_queries == b.n_queries`. It does **not** verify
that the two representations came from the same query draw, the same layers, the
same pooling, or the same activation mode. Matching row counts is not matching
semantics, and two 64-row matrices from different draws will compare happily and
mean nothing.

Guard this at the level above: `_functional_matrix` resolves one draw for the
whole collection and refuses when several are present, and the `functional_repr`
availability token is only exact when `scan_cache` is given a draw.
