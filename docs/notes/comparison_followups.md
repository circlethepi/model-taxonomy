# Follow-ups to the cross-taxonomy comparison

The comparison machinery landed as a library. Three pieces were deliberately left
out of that pass to keep it focused; this is what they are and where they hook in.

## What exists now

```python
from src.analysis import scan_cache, compare_all_slices

index = scan_cache("results/shared_cache").with_available(
    "structural_weights", "dataset_embedding"
)
compare_all_slices(
    index,
    taxonomies={"structural": "cosine", "dataset_embedding": "cosine"},
    output_dir="results/<experiment>/comparison",
)
```

- `src/analysis/ground_truth.py` — recipe → mixture components → regular simplex,
  plus `pure_anchors` / `evaluation_points`.
- `src/analysis/discovery.py` — `scan_cache` joins adapters to recipes on
  `recipe_hash`; `CacheIndex.slices(by=...)` produces the groupings.
- `src/analysis/comparison.py` — `build_taxonomy_artifacts` (cache-backed distances
  and embeddings), `compare_taxonomies`, `compare_all_slices`, `TaxonomyComparison`.
- `src/cache/collection_cache.py` — `index.json`, and geometries keyed by
  `{method}_{n_components}d`.

Reports land in `results/<experiment>/comparison/{grouping}/{slice}/` as
`report.json` + `report.md`, with `ground_truth/`, `distance_matrices/`,
`geometries/` and `simplex/` beside them.

---

## 1. `scripts/run_comparison.py` and YAML integration

**Goal:** one experiment YAML drives the whole pipeline, comparison included.

```
python scripts/run_comparison.py experiments/foo.yaml --from-cache
python scripts/run_comparison.py experiments/foo.yaml --compute
```

`--from-cache` reads whatever `CollectionCache` already holds and refuses to
compute; `--compute` fills gaps. Both are thin wrappers — `compare_all_slices`
already does cache-hit-then-compute per taxonomy, so the script is argument
parsing plus config plumbing.

Proposed YAML block:

```yaml
comparison:
  taxonomies:                 # taxonomy -> metric
    structural: cosine
    dataset_embedding: cosine
  structural:
    layers: [27]              # forwarded to build_taxonomy_artifacts
    projections: o
  groupings:                  # default: all four
    - [n_samples, seed]
    - [n_samples]
    - [seed]
    - []
  n_permutations: 999
```

`taxonomy.n_components` already exists and accepts a scalar or a list
(`scripts/_utils.geometry_dims`), so nothing new is needed for the dimension.

Then register the step in `scripts/run_experiment.py` and `scripts/submit_slurm.py`
alongside the existing steps.

**One thing to decide first:** `build_taxonomy_artifacts` takes a `CacheIndex`,
which is discovered from the cache, whereas the rest of the pipeline resolves
models from the YAML via `scripts/_utils.resolve_model_ids`. Either teach the
script to build a `CacheIndex` from resolved model IDs, or let `comparison.models`
accept the same `base_models` / `fine_tuned` tokens the other sections use. The
second is more consistent with the existing configs.

## 2. Visualization

Nothing plots the new objects yet. `src/plots/figures.py` has `plot_scatter`
(2-D `GeometryResult`), `plot_distance_heatmap` and `plot_lines`, which cover the
inputs but none of the outputs.

Wanted, as `src/plots/simplex.py`:

- **Ternary / simplex plot** — the `k=3` case drawn as a triangle with each model
  at its barycentric position, true position and recovered position joined by a
  segment. For `k=2` this degenerates to a 1-D strip, which is still the clearest
  view of the current yahoo data.
- **Recovered vs. true scatter** — one panel per taxonomy, `y = x` reference. This
  is the figure that distinguishes "monotone but non-linear" from "just wrong", and
  `RecoveryResult.pairs()` already returns exactly the right triples.
- **Cross-taxonomy disparity heatmap** — `TaxonomyComparison.pairwise_procrustes`
  as a matrix. Can reuse `plot_distance_heatmap` by wrapping with
  `as_distance_matrix`, as `correlation_table`'s docstring suggests.
- **Shepard diagram** — `src/analysis/quality.py::shepard` returns the data; there
  is no plotter. Needed before any 2-D scatter is trusted, and it is the same
  object note 2 of `distance_matrix_comparison.md` wants to fit.
- **Convergence-in-`n` line plot** — read `report.json` across
  `by_seed/`, plot mean L1 / Procrustes disparity against `n`, one line per
  taxonomy. Directly serves the convergence question.

Plus a driving notebook (`notebooks/4_taxonomy_comparison.ipynb`).

Load the `dataviz` skill before writing any of these — the repo has a palette and
style in `src/plots/config.py` to stay consistent with.

## 3. `docs/guides/pipeline_stages.md`

A runnable template YAML per stage, so a single stage can be run in isolation:

1. create datasets
2. generate dataset embeddings / dataset taxonomy
3. fine-tune on datasets
4. generate structural representations / structural taxonomy
5. create the common test dataset
6. inference → functional representations / functional taxonomy
7. inference → behavioral representations / behavioral taxonomy
8. taxonomic comparison

Each entry wants: which script runs it, the minimal YAML, what it writes to the
shared cache, and what it needs to already be there. `experiments/example.yaml`
and the `yahoo_topics_mean_cosine*` family are the sources to factor from. Worth
writing **after** the cache renumbering (`cache_layout_migration.md`), since the
"what it writes" column should quote the final directory names.

---

## Smaller items noticed in passing

- **`functional` and `behavioral` have no cached representations at all** —
  `results/shared_cache/representations/` is empty, so `build_taxonomy_artifacts`
  raises `NotImplementedError` with a pointer to `extract_reprs.py` for those two
  levels. Only structural and dataset_embedding are comparable today.
- **`src/metrics/vector.py` carries a TODO** (also in
  `memory/project_full_pipeline_metrics.md`): it assumes a uniform 2-D
  representation matrix and will break on structural reps once row truncation is
  removed. The comparison layer sidesteps this by routing structural through the
  low-rank builders, so it is not urgent, but the full-pipeline structural path
  still needs it.
- **`torch` costs ~400 MB of RSS purely as an import**, because
  `src/notebook/lora_weights.py` opens adapters with `safe_open(framework="pt")`
  and then calls `.numpy()`. The adapter weights are `float32`, so
  `framework="numpy"` would return the same arrays and drop torch from the
  dependency path of every distance computation — measured 587 MB → ~190 MB of
  import overhead. Cheap win; check for `bfloat16` adapters first, since numpy has
  no such dtype.
- **`compare_taxonomies` projects from the largest available dimension, not
  `k-1`.** Projecting from exactly `k-1` makes the anchors' affine hull the whole
  space, so every residual is identically zero and the off-simplex diagnostic
  disappears. `scripts/check_analysis.py::t_projection_dimension_matters` pins
  this. Worth remembering before anyone "simplifies" it back.
