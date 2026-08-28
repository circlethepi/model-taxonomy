# Cross-Level Comparison

How a taxonomy's answer is scored against a known ground truth, and how the levels are
compared with each other. Everything here lives in `src.analysis`.

## Fleet-level surrogate transforms

A `DistanceMetric` sees exactly two models, so it can never subtract a fleet mean or
divide by a fleet covariance. Those are **collection**-level operations: they are defined
by the set of models being compared and change when a model joins or leaves it.
`src/analysis/surrogates.py` is where they live, between representation resolution and
distancing.

### Why the levels need them

Every representation carries a large component shared by every model in the collection,
which by construction encodes nothing about what distinguishes them:

- **dataset** — a recipe's centroid is dominated by "this is Yahoo answer text". Cosine
  distances across the 16 simplex3 recipes span 0.00–0.03, so the mixture geometry is a
  small perturbation riding on one shared direction.
- **behavioral** — all 16 models answer the same 100 questions, so most of a generation
  embedding is question content, identical across models by design.
- **functional** — hidden states are dominated by the base model's own representation of
  the prompt; LoRA moves them a little.

Centering removes that component; what is left is the between-model variation, which is
what a taxonomy is trying to measure. This changes what cosine *means* — on centered rows
it is a correlation rather than an angle to the origin — which is why it is an explicit,
recorded transform rather than a silent default.

### `center_representations(reps, mode="grand" | "rowwise")`

| Mode | Subtracts | Requires |
|---|---|---|
| `"grand"` | one `(1, d)` vector: the mean over every row of every model | nothing — the only mode available for a `(1, d)` pooled representation, and the safe one whenever row counts differ |
| `"rowwise"` | the per-row fleet mean: row *i* minus the average of row *i* across models | every model to have the same rows in the same order |

`"rowwise"` is the stronger transform where it applies, because it removes the *query's*
contribution rather than a global average of it — for the behavioral level that is the
question text.

The grand mean is weighted by **rows, not by model**: an unweighted average of per-model
means would be a different vector whenever row counts differ, and would give a model with
100 rows the same say as one with 1600. Row counts are preserved, so the result stays
usable with the row-aligned metrics (Frobenius, CKA) as well as the permutation-invariant
ones.

### `whiten_representations(reps, shrinkage=0.1, mode="grand")`

Centers, then decorrelates against the pooled fleet covariance: `X → (X − μ) S^{-1/2}`,
with `S` shrunk toward a scaled identity, `S_shrunk = (1 − a)S + a(tr S / d)I`.

*shrinkage* is not optional and defaults to a deliberately non-negligible `0.1`. With 16
models at d=768 the pooled covariance is badly conditioned in exactly the low-variance
directions, and `S^{-1/2}` without shrinkage multiplies those directions — which are
estimation noise — by the largest factors in the whole map. The result looks like
structure and is not. `a=0` is accepted but raises if the covariance is singular.

Whitening equalizes variance across directions, so a small consistent between-model
difference counts as much as a large one. That is the point when the large directions are
shared nuisance and a liability when they are real signal — so it is offered *alongside*
centering rather than instead of it.

### Using them

`centered(mode=...)` and `whitened(shrinkage=..., mode=...)` return a `Transform` callable
suitable for the `transform=` argument of `build_taxonomy_artifacts` and
`resolve_ordered`. `transform_key(transform)` gives the short string that identifies one
in a cache handle or a figure label.

Every function returns fresh `ModelRepresentation` objects and records what it did in
`metadata["surrogate_transform"]`, so a figure panel can always say which surrogate it is
showing.

## Scoring against the ground truth

The simplex3 experiments know where every model *should* be: its mixture over the three
topic groups. `src/analysis/ground_truth.py` builds that truth and scores a taxonomy
against it two ways, which do not measure the same thing.

| | `dcor_vs_truth(dm, truth_dm)` | `disparity_vs_truth(dm, truth_geometry, ...)` |
|---|---|---|
| Scores | the **distances** | the **configuration** — the arrangement an embedding actually draws |
| Range | bias-corrected distance correlation | `[0, 1]` |
| Direction | **higher is better** | **lower is better** (0 = identical shape) |
| Embeds? | no — untouched by MDS distortion | yes — MDS-mediated, inherits its distortion |

They come apart: a taxonomy can reproduce the pairwise distance profile while arranging
the points in something that is not the simplex. Report them side by side, and say which
way each runs — they run in opposite directions.

**`dcor_vs_truth`** is invariant to a constant rescaling of either matrix, which is why
`simplex_distance_matrix` and a plain `pdist` over the raw weight vectors — which differ
by exactly a factor of `1/√2` — score identically. The U-centred dCor\* lives on a squared
scale and may legitimately be negative; **do not clip it**.

**`disparity_vs_truth`** is the *scaled residual* Procrustes disparity: the residual sum
of squares as a fraction of total squared coordinate variance. Because it is
MDS-mediated, read it beside `kruskal_stress`, not instead of it. For the same reason
`random_state` is an explicit argument rather than a hidden default: `MDSGeometry`
initialises randomly, so a caller wanting this score to describe the configuration it is
*plotting* must pass the seed that configuration was fitted under. Pass an already-fitted
`geometry=` to score an embedding you have in hand instead of fitting a second one — `dm`
is then unused, and it is on the caller to pass a geometry that actually came from it.

Models are paired by identifier, not by row position: `procrustes_compare` reindexes both
configurations onto their common `model_ids` first. So permuting rows together with their
ids leaves the score unchanged, and mislabelled rows are a genuine disagreement rather
than a silent one.

## Distributional distance metrics

`src/metrics/distributional.py` adds `EnergyDistanceMetric` and `MMDDistanceMetric`.

The metrics in `frobenius.py` and `cka.py` compare two **indexed lists of vectors**: row
*i* of one model is assumed to mean the same thing as row *i* of the other, and permuting
one input changes the answer. That assumption is right when rows are queries in a shared
draw, and wrong as soon as they are not.

These two treat each representation as a **sample from a distribution** over the feature
space, so they are invariant to row order and tolerant of unequal row counts — the same
property `BuresWassersteinDistanceMetric` has, but without its restriction to second
moments. BW compares two clouds through their covariances alone, which cannot see a
difference in shape at equal covariance; MMD with a characteristic kernel and the energy
distance both can.

**Where this matters here:** the behavioral level stores `(n_queries × replicates, d)` in
query-major order. Under sampling, the *spread* of a model's 16 replicates for one
question is part of what distinguishes it, and pairing replicate 3 of model A with
replicate 3 of model B is meaningless — they are independent draws. A distributional
distance is the honest reading of that matrix.

Both estimators cost `O(n_a · n_b · d)` per pair, dominated by the cross Gram. At the
sizes here (1600 behavioral rows, 1000 dataset rows) that is well under a second per pair.

Both **require more than one row** and reject kernel-matrix representations, with an
error naming `representation="matrix"` as the fix.

## Comparing whole collections

`build_taxonomy_artifacts(index, taxonomy, metric, ..., transform=None)` returns the
distance matrix plus embeddings for one taxonomy over one collection, reading from and
writing back to `CollectionCache`.

`resolve_ordered(index, taxonomy, ids, ..., transform=None)` returns
`(reps, order)` — the representations for *ids*, in *ids* order, with *transform* applied.
It is split out of the distance-matrix path so that **a sweep over metrics at one selector
resolves once**. That is the shape of every panel grid: a row fixes the selector and the
columns vary the metric, and going back through the full path per column re-read the same
tensors from disk. `order` is the permutation into `index.entries`; structural needs it
because it reads the adapter files itself and so returns `reps is None`.

`compare_taxonomies(distance_matrices, recipes, ...)` runs every comparison over one
collection — Mantel, dCor, Procrustes, PROTEST, simplex recovery. Model sets may differ;
the intersection is used. `recipes` is where the ground truth comes from.

`compare_all_slices(index, taxonomies, output_dir, ...)` compares every slice of a
collection and the pooled whole, writing reports. The four default groupings answer
different questions from the same cache: `(n_samples, seed)` isolates one experimental
cell, `(n_samples,)` varies the seed within a size, `(seed,)` varies the size within a
seed — the axis a convergence-in-`n` study needs — and `()` pools everything. A slice that
cannot be compared is recorded in the manifest with the reason and skipped, rather than
aborting the sweep.

> **Mantel's p-value is descriptive only.** `distance_correlation` / `dcor_test` are the
> stronger test and `compare_taxonomies` reports `dcor_vs_truth` beside
> `matrix_corr_vs_truth`. Mantel's statistic is untouched; its p-value now carries a
> warning.

## See also

- [Visualization](visualization.md) — the figures and agreement tables these scores feed
- [Core Concepts](../concepts.md#analysing-the-results) — the containers everything here
  operates on
- [API Reference](../api_reference.md#analysis) — full signatures
