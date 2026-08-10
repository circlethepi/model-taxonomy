# Comparing distance matrices, and mapping one onto the simplex

Two open questions bookmarked during the cross-taxonomy comparison work. Neither
blocks anything that ships today; both change what the comparison *reports*.

---

## 1. Mantel is the weakest test we run, and PROTEST is already better

> **Done (TODO item 6).** `distance_correlation` and `dcor_test` are in
> `src/analysis/matrices.py`, `compare_taxonomies` reports `dcor_vs_truth`
> beside `matrix_corr_vs_truth`, and Mantel's p-value now carries a warning
> while its statistic is untouched. Read "What the implementation found"
> at the end of this section before using the numbers — two of the findings
> change how the output should be read, and neither was anticipated here.

The concern is real and well documented. `mantel_test`
(`src/analysis/matrices.py`) correlates the two matrices' off-diagonal vectors and
permutes the row/column index to build a null. The problem is that the
`n(n-1)/2` off-diagonal entries are **not independent** — they are derived from
`n` points, so every entry shares a point with `n-2` others. When the underlying
distances are spatially autocorrelated, which geometric distances between related
models certainly are, that dependence inflates the test statistic's variance and
the permutation null does not absorb it. The literature reports inflated type-I
error (rejecting "unrelated" too often) for exactly this reason.

**The practical answer is that we already have the better test.**
`protest` in `src/analysis/configurations.py` is a Procrustes-based permutation
test on the same null (permute which model is which, re-superimpose). Peres-Neto
& Jackson (2001), *"How well do multivariate data sets match? The advantages of a
Procrustean superimposition approach over the Mantel test"*, compared them
directly and found PROTEST has better power and better-calibrated type-I error.
It works on configurations rather than matrices, which is a real difference — it
needs an embedding first, and so inherits that embedding's distortion (which is
what `kruskal_stress` is for).

`compare_taxonomies` reports both, per taxonomy:
`matrix_corr_vs_truth` (Mantel-style correlation) and `protest_p_value`.

### Candidates worth evaluating before changing the default

| method | what it gives | why consider it |
|---|---|---|
| **PROTEST** (have it) | permutation p-value on Procrustes disparity | better calibrated than Mantel; already implemented |
| **Distance correlation (dCor)** | dependence measure, zero iff independent | works directly on distance matrices, no embedding step, and unlike Mantel it detects non-monotone dependence |
| **RV coefficient** | matrix-level correlation of configurations | the classical multivariate analogue; closely related to Procrustes |
| **dbRDA / PERMANOVA** | variance in one matrix explained by the other | natural if the mixture proportion is treated as a predictor rather than a second geometry |
| **Mantel + Freedman–Lane** | Mantel with a residual permutation scheme | keeps the matrix-level framing while improving calibration; relevant if a partial/conditional version is ever needed |

Suggested next step: keep reporting the Mantel-style correlation as a descriptive
number (it is cheap and interpretable), stop treating its p-value as evidence, and
add dCor as the matrix-level test since it needs no embedding and so is
independent of the MDS step. Do **not** silently drop Mantel — existing figures
reference it.

### What the implementation found

Three things, of which the first two change how the output must be read.

**1. dCor is unsigned, so it does not subsume the correlation.** The plan above
treats dCor as a strictly better matrix-level statistic. It is not a
*replacement*: it measures dependence, so a taxonomy that recovers the mixing
order exactly **backwards** scores `dCor = 1.0`, identically to a perfect one
(verified on the 1-D truth matrix). That is not a hypothetical — the behavioral
level does exactly this, and it is the whole content of the 2026-08-05 table in
`TODO.md`. So the reason to keep Mantel is stronger than "existing figures
reference it": the **sign** is information dCor structurally cannot carry.
Pinned by `t_dcor_unsigned`.

**2. At five models, no matrix-level test can reach p < 0.05, and that is the
design's fault rather than the data's.** With `n = 5` there are only 120
relabellings, which would put the floor at 1/120 ≈ 0.008 — but the ground truth
on these slices is five evenly-spaced points on a 1-D simplex, and that matrix
is symmetric enough that **8** of the 120 relabellings attain the maximum. A
taxonomy reproducing the truth *exactly* therefore scores `p = 8/120 ≈ 0.067`,
and the entire null takes **4 distinct values**. Read the p-values as ordering
evidence, never against a 0.05 threshold. This is an argument for more adapters
per slice, and it applies to PROTEST's permutation p-value for the same reason.
Pinned by `t_dcor_permutation_floor`.

**3. The bias correction is not optional at this scale.** Measured over 2,000
draws of two *independent* five-model matrices, the classical V-statistic dCor
averages **0.846**; the U-centred `dCor*` of Székely & Rizzo (2013) averages
**−0.007**. Hence `bias_corrected=True` is the default. The cost is that `dCor*`
lives on the squared scale and may be negative.

**Caveat carried into the docstring:** `dCor = 0` characterises independence only
for metrics of strong negative type (Lyons 2013). Euclidean qualifies; the cosine
and CKA distances used here are not known to.

### First numbers on the real slice

Same slice as everything else — 5 adapters, `n_samples=1000, seed=0`, cosine —
with the two existing columns for comparison:

| taxonomy | matrix corr | PROTEST p | dCor* | dCor p |
|---|---|---|---|---|
| dataset_embedding | 1.0000 | 0.005 | 0.9667 | 0.0667 |
| functional | 1.0000 | 0.015 | 0.9285 | 0.0667 |
| structural | 1.0000 | 0.754 → see note | 0.7537 | 0.1000 |
| behavioral | −1.0000 | 0.955 | 0.3870 | 0.2667 |

Three of the four sit at the 0.0667 floor, so the p-values rank the levels but
do not test them. The informative column is `dCor*` itself: it separates
dataset_embedding and functional (≈0.93–0.97) from structural (0.75) and
behavioral (0.39) — and note that behavioral's 0.39 is *unsigned*, so it is
weak dependence, not merely inverted dependence. A perfect reversal would have
scored 1.0.

---

## 2. A learned map from taxonomy distances to the simplex, and whether it converges

The idea: rather than only *scoring* how close a taxonomy's geometry is to the
ground-truth simplex, find a transformation

```
T : D_taxonomy  ->  D_simplex
```

and ask whether `T` converges to a fixed object as the number of samples in each
dataset grows. If it does, that limit is a meaningful statement about the
taxonomy: it says the taxonomy measures the recipe simplex up to a specific,
stable distortion.

### The data for this already exists

`compare_all_slices` writes one report per slice across four groupings, and the
`by_seed/` grouping varies `n_samples` within a fixed seed — which is exactly the
axis this needs. Run against the current cache
(20 adapters, n ∈ {1, 10, 100, 1000}, seed 00, structural + dataset_embedding at
cosine), the per-`n` numbers already show the phenomenon:

| n | taxonomy | Spearman rho | mean L1 | max residual | Procrustes vs truth |
|---|---|---|---|---|---|
| 1 | dataset_embedding | 0.866 | 0.667 | 0.001 | 0.250 |
| 10 | dataset_embedding | 1.000 | 0.205 | 0.105 | 0.279 |
| 100 | dataset_embedding | 1.000 | 0.057 | 0.000 | 0.004 |
| 1000 | dataset_embedding | 1.000 | 0.054 | 0.000 | 0.004 |
| 1 | structural | 1.000 | 0.667 | 0.001 | 0.250 |
| 10 | structural | 1.000 | 0.550 | 0.866 | 0.536 |
| 100 | structural | 1.000 | 0.581 | 0.571 | 0.501 |
| 1000 | structural | 1.000 | 0.501 | 0.533 | 0.450 |

`dataset_embedding` converges — mean L1 falls 0.667 → 0.054 and appears to have
settled by n=100. `structural` does not: its *ordering* is perfect at every size
(rho = 1.000) while its spacing stays wrong (mean L1 ≈ 0.5) and it sits far off
the simplex (residual ≈ 0.5). That is a substantive difference between the two
levels, and it is the reason to want `T` rather than a single score: the
structural geometry appears to be a *monotone but strongly non-linear* function of
the mixture, which a scalar disparity cannot express.

Caveats on the table above: five models per slice, one seed, one
`(layer, projection)` block (27, `o`), and `n=1` means one text per dataset, so
that row is close to noise. Treat the shape of the trend, not the values.

### A first `T_n` already exists, for free

`compare_taxonomies` fits a Procrustes similarity map from each taxonomy's
embedding onto the ground-truth simplex and now **persists it** — rotation,
scale, and the centroid/norm that were divided out — to
`{slice}/procrustes/vs_truth/{taxonomy}/`, with
`ProcrustesResult.transform(coords)` to apply it. The direction is
taxonomy → ground-truth frame.

That is a restricted `T_n` (similarity transforms only, on coordinates rather than
distances), but it is enough to run the convergence experiment described below
without writing any new fitting code: load the map at each `n`, apply all of them
to one common set of coordinates, and measure how far apart consecutive maps
land. Its limitation is exactly the interesting one — a similarity transform
cannot express the monotone non-linearity that `structural` shows, which is the
argument for fitting an isotonic or parametric `T` on distances instead.

Measured on the current cache at `n=1000, s00`: `dataset_embedding` has
`scale = 0.998` and maps to within a mean of 0.012 of the truth;
`structural` has `scale = 0.742` and a mean deviation of 0.187.

### Candidate forms for T

- **Isotonic / monotone regression** on the off-diagonal pairs. This is the
  Shepard diagram (`src/analysis/quality.py::shepard`) promoted from a diagnostic
  to a fitted object, and it directly matches the "monotone but non-linear"
  behaviour structural shows. Non-metric MDS already fits such a transform
  internally — `MDSGeometry(metric=False)` — so a first cheap experiment is to
  compare metric and non-metric MDS at each `n`.
- **Parametric** (`d -> a·d^b`, or a low-order polynomial). Fewer degrees of
  freedom, so convergence in `n` becomes convergence of two or three numbers,
  which is much easier to plot and reason about than convergence of a function.
- **Linear on the Gram matrix** — equivalent to asking whether one configuration
  is an affine image of the other, which is what `procrustes_compare` tests under
  a restriction to similarity transforms. Relaxing to general affine would make
  the shear that `simplex.py` warns about (its barycentric guarantee is only
  affine-invariant *within* the hull) an explicit part of the model.

### What convergence would need to mean

"Converges as `n` grows" needs a metric on transforms before it can be measured.
The cheapest defensible version: fit `T_n` at each sample size, then report
`sup |T_n(d) - T_{n'}(d)|` over the observed distance range for consecutive sizes,
and check it shrinks. That reduces the question to a single decreasing sequence
per taxonomy, which is plottable and testable against the existing per-slice
reports.

**Identifiability caveat.** MDS fixes a configuration only up to rotation,
reflection, translation and uniform scale, so `T` is only identifiable once that
freedom is quotiented out — which is why the comparison layer works in barycentric
coordinates and Procrustes disparities in the first place. Fit `T` on
**distance matrices** (invariant) rather than on coordinates, or on
Procrustes-aligned coordinates, but not on raw MDS output.
