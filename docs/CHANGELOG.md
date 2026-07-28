# Changelog

<!-- When opening a PR, always add an entry under "Unreleased" describing what changed and why. -->

## Unreleased

### New package `src/analysis` — analysis over distance matrices and geometries

Until now the repo produced distance matrices in two places that could not talk to
each other: the config-driven pipeline, which yields a typed `DistanceMatrix`, and
the low-rank LoRA builders in `src/notebook/structure.py`, which yield a bare
`(names, ndarray)` tuple and were never persisted. Everything downstream —
`src/plots`, `CollectionCache`, `GeometryResult` — only spoke the first. This
package makes both feed one analysis layer.

**`src/analysis/bridge.py`**

- `as_distance_matrix()` wraps a `(names, matrix)` pair as a `DistanceMatrix`,
  converting similarity to distance when asked.
- `lora_distance_matrix()` dispatches to the four low-rank builders in
  `src/notebook/structure.py` and returns a `DistanceMatrix`.
- `fit_geometry()` mirrors `scripts/_utils.make_geometry` but leaves
  `n_components` free — that factory hardcodes 2, which ruled out 1-D embeddings
  and simplex projection in the full embedding dimension.
- `save_collection()` persists a notebook-built matrix through `CollectionCache`.

**`src/analysis/matrices.py`** — distance-matrix level. `match_models()` (set
intersection + reindexing, needed because taxonomy levels cover different model
sets in different orders), `offdiag()`, `matrix_correlation()`, `mantel_test()`,
`correlation_table()`.

**`src/analysis/configurations.py`** — point-configuration level, all Procrustes
based and therefore invariant to the rotation/reflection/scale an embedding picks
arbitrarily. `procrustes_compare()` (disparity matches `scipy.spatial.procrustes`),
`per_point_residuals()`, `protest()`, `align_to_reference()`, `point_dispersion()`.

**`src/analysis/quality.py`** — embedding fidelity. `kruskal_stress()` recomputed
from coordinates so PCA and UMAP results are comparable with MDS (only MDS
populates `GeometryResult.stress`), and `shepard()` for the per-pair view.

**`src/analysis/simplex.py`** — barycentric projection onto a simplex of anchor
models, plus `compare_simplices()` and `anchor_weight_vs_truth()`. The last is the
only check in the package that compares a geometry against a quantity external to
the pipeline — the known data-mixing proportion — rather than to another derived
quantity.

**`src/notebook/structure.py`**

- `plot_distance_analysis()` now embeds via `MDSGeometry` instead of calling
  sklearn directly with `dissimilarity="precomputed"`, which is deprecated in the
  pinned scikit-learn 1.9. One MDS code path in the repo now.

**`scripts/check_analysis.py`** (new) — runnable verification: 19 synthetic checks
plus 4 against the real adapter cache. Deliberately low-rank throughout; the
cosine identity is proved on small synthetic factors because it is algebraic and
dimension-independent.

### Package exports brought up to date

- `src/taxonomy/__init__.py` exported only `BehavioralTaxonomy`; the other four
  taxonomies are now exported too.
- `src/metrics/__init__.py` now exports `CosineDistanceMetric` and
  `DotProductDistanceMetric` from `vector.py`, which were reachable only via
  `scripts/_utils.make_metric`.
- `src/cache/__init__.py` now exports `DatasetEmbeddingCache` and
  `SampledDatasetCache`.

### Docs corrected

`docs/api_reference.md`, `docs/guides/structural_taxonomy.md` and
`docs/concepts.md` still documented the `n_components` truncation removed in
`cbc1579` and the pre-`{config_hash}` `LoRACache` layout. Updated to the real
signatures, the `(N_layers, max_len)` zero-padded representation shape, and the
`extraction_config` parameter. `docs/guides/extending.md` now says explicitly that
its `StructuralTaxonomy` sketch is a teaching example, not the real class.

### LoRACache — multi-config storage + adapter listing

**`src/cache/lora_cache.py`**

- New storage layout: extracted representations now live in a `{config_hash}/` subdir
  under each adapter folder, so multiple extraction configurations can coexist.
  Raw PEFT files (`adapter_model.safetensors`, `adapter_config.json`) are untouched.
- `exists()`, `load()`, `load_config()`, `save()` now take an `extraction_config: dict`
  parameter to identify which config-hash subdir to use.
- `save()` now accepts and stores `layer_lengths: list[int]` in `config.json`.
- New methods: `list_raw_adapters()`, `adapter_status()`, `adapter_count()`,
  `list_representations()`.
- `list_adapters()` updated to detect processed adapters via config-hash subdirs.

**`src/taxonomy/structural.py`**

- Removed `n_components` parameter and `_truncate_pad` — LoRA weight vectors are
  now stored at their full natural length. Rows of different length are zero-padded
  to the longest so the matrix can be stacked; original lengths saved in `config.json`.
- Added shorthand layer/projection selectors: `layer_indices` and `projections`
  (matching `load_lora_weights` conventions). `layer_names` still works for explicit
  full-prefix control and takes precedence.
- `_find_lora_pairs` updated to apply the new filters via architecture-agnostic
  regex on actual parameter names.
- `extract()` now builds a single `extraction_config` dict and passes it to all
  `LoRACache` calls so the right config-hash subdir is always targeted.

**`src/metrics/cka.py`, `src/metrics/frobenius.py`**

- Added TODO comments noting that these metrics assume uniform-row
  `ModelRepresentation.matrix` and will need updating before structural taxonomy
  representations can flow through the full pipeline.
