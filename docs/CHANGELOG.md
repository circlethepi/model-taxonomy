# Changelog

<!-- When opening a PR, always add an entry under "Unreleased" describing what changed and why. -->

## Unreleased

### The behavioral cache was orphaned, and is now keyed like the functional one

`05_generated` was keyed `{config_hash}/` with per-model filenames hashing the
adapter's **full path**. The stored paths were *relative*, so the key depended on
the working directory the extraction ran in. This was not a latent hazard:
measured before the change, `behavioral_repr` resolved to **0 hits across all 25
adapters × 2 configs** while 10 representations sat readable on disk. Every write
had succeeded; the cache read as empty.

It went unnoticed because both checks touching behavioral bypassed the broken
path — one read by config hash directly, the other excluded behavioral unless
exactly one config existed, and there were two. **The level had never been
exercised end to end.** It now is, and immediately produced its first
cross-taxonomy numbers (see `docs/notes/TODO.md` — they are inverted, which is
what TODO item 11 predicts).

**`src/cache/_draw_keyed.py` (new)** — `DrawKeyedCache`, the addressing half both
inference caches now share: `{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/`,
`draw_name`/`draw_dir`/`mode_token`/`config_hash`, the surrogate mechanism, and
the enumeration API. Extracted as a pure move first, gated on an unchanged
suite. `t_draw_keyed_shared_key` asserts the two caches share these by
**identity**, so a future override cannot silently re-diverge them.

**`src/cache/generated_text_cache.py`** — rebased onto it. Variants live in the
filename (`generation{max_new_tokens}_{embedder_hash}`), so `ls embeddings/`
answers "which embedders ran over this draw?" and re-embedding existing text
costs no GPU pass. `model_slug` is deleted; its one live use was an in-memory
PEFT adapter name, now `_hf_inference._adapter_name`, where hashing a path is
harmless because the name never outlives the process.

**No query text is stored at either level.** `(recipe_hash, n_samples, seed)`
determines it, because `text_field` is part of the recipe and so part of
`recipe_hash`. `_replay_queries` stopped guessing the column from a candidate
list — first-match-wins silently took `text` over `question_title` on rows
carrying both — and reads `text_field` from `recipe.json`. `source_indices` is
now populated rather than always empty.

**Consumers** — `scan_cache` takes `behavioral_draw` (the old keyword is removed,
not aliased, so a stale call fails loudly); `_functional_repr_exists` collapses
into `_draw_keyed_repr_exists`; `_behavioral_matrix` mirrors `_functional_matrix`
behind `behavioral_selector`; `compare_all_slices` forwards the selectors it
never had.

**Migration** — `scripts/migrate_behavioral_layout.py`, with
`--dry-run`/`--apply`/`--revert`. All 10 entries round-tripped byte-identical;
the payoff check reports 5 model-draws sharing coordinates with `04_activations`
and 0 for the other draw, as predicted. 128 stored query strings were dropped as
redundant; generated text is preserved in full.

Checks **54 → 60 passed, 0 failed, 0 skipped**.

### Functional taxonomy: `ActivationCache`, read-time views, and a redefined `gram`

The functional level was the last one still writing through the flat `DiskCache`,
whose key hashes every query string into one opaque filename. It now has its own
cache, a comparison-layer path, a discovery token, and checks at all three tiers.

**`src/cache/activation_cache.py` (new)** — `ActivationCache`, keyed **model-wise
then draw-wise** (`04_activations/{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/`),
following `LoRACache` rather than `GeneratedTextCache`. One forward pass produces
every hidden state, so extraction stores all of them — one file per
`(mode, pooling, layer)` — and *which layers you compare* becomes a read-time
choice rather than a reason to re-run inference. Writes are additive: adding a
mode or a layer never rewrites an existing file. Views are computed lazily and
written back under `surrogates/{hash}/`. `queries.json` stores source row indices,
not text, since `01_datasets` is canonical.

**`src/taxonomy/_hf_inference.py` (new)** — `HFInferenceTaxonomy`, the base-model
reuse and adapter-swapping machinery extracted from `BehavioralTaxonomy` without
behavioural change. Both inference levels now inherit it, so the left-padding pin
cannot be set for one and forgotten for the other.

**⚠️ Breaking: `gram` has been redefined.** It is now `G = H Hᵀ` of the
concatenated feature matrix — `(n_queries, n_queries)`, **rows are queries**. It
used to be stacked upper triangles of per-layer Gram matrices, `(n_layers,
n_queries(n_queries+1)/2)`, **rows were layers**. These are different objects; the
old form is gone rather than kept as an option. Nothing needs migrating only
because the functional level had never been run. The `representation:` config key
is replaced by `view: concat | gram`. See `docs/notes/gram_and_cka.md`.

**Mask-aware pooling.** `FunctionalTaxonomy._pool` now takes the `attention_mask`
and pools only over real positions. Previously it averaged over the padded length,
which made a pooled vector depend on which other queries shared its batch — and
the cache could not notice, since `batch_size` is not part of the key.

**`CKADistanceMetric` raises instead of returning NaN.** The unbiased HSIC
estimator divides by `n(n-3)`; below 4 rows it now raises and names
`unbiased=False`. It also refuses representations tagged `metadata["is_kernel"]`,
which is what stops a stored Gram from silently being turned into `(H Hᵀ)²`.

**Layerwise normalization, and it is the new default.**
`normalize_activations` / `ActivationCache.load(normalize=)` widen from a bool to
`"layer" | "global" | "none"`; bools still work (`True → "layer"`,
`False → "none"`) and are canonicalized before hashing so one request cannot
become two surrogates. `"layer"` row-normalizes each `(mode, layer)` block before
concatenating, then normalizes the row, so rows stay unit-norm and `gram`'s
diagonal stays 1. `"global"` — concatenate, then normalize once — is the previous
behaviour, unchanged and still reachable by name.

⚠️ **This changes the numbers.** Any `concat` or `gram` view is a different
matrix under the new default, and `{"normalize": true}` no longer hashes to the
same surrogate as `{"normalize": "layer"}`, so surrogates written before this
change are orphaned and rebuilt on next read (cheap — a concat over stored
activations, no model). Stored per-layer activations are untouched.

Measured on Llama-3.2-3B, mean-pooled: the transformer blocks' row norms sit
within ~1.6× of each other, but the **embedding layer and layer 1 are two orders
of magnitude smaller** (0.36 and 2.41 against 56–71), so under `global` they
carry 0.00% and 0.01% of a row and are effectively absent from a comparison
labelled "all layers". Under `layer` all 29 count equally. On the smoke run the
two modes correlate at r = 0.993 — a change in emphasis, not a different
measurement.

**Also:** `_functional_matrix` + `functional_selector` in
`src/analysis/comparison.py`; a `functional_repr` availability token (flag letter
`F`, header now `WRDSBF`) and `functional_draw` in `scan_cache`;
`make_activation_cache` in `scripts/_utils.py`, with `REPR_CACHE_DIRS` and
`make_repr_cache` removed as dead; `experiments/yahoo_functional_smoke.yaml`;
and seven synthetic, one `[data]` and one `[gpu]` check.

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

**`src/analysis/identity.py`** — reconciles the identifier schemes different
taxonomy levels use for the same objects. `DatasetEmbeddingTaxonomy` keys by
recipe ID (`yahoo_topic0_only`) while the model-level taxonomies key by adapter
path (`.../yahoo_topic0_only_r16`), so a set intersection between them returned
nothing and `dataset_embedding` was reported as incomparable in every
cross-taxonomy table. `recipe_id_for()` maps an adapter to the recipe it was
trained on, reading the `dataset_name` that `finetune_lora.py` records in
`experiment_meta.json` (falling back to parsing the directory name only when
that file is absent), and returns non-adapter identifiers unchanged.
`relabel()` rewrites an object's `model_ids`, refusing any rewrite that would
collide two distinct models onto one identifier — which is what a rank or
init-seed sweep would do. `id_overlap()` reports why a comparison found nothing
in common.

All the comparison functions (`match_models`, `matrix_correlation`,
`mantel_test`, `correlation_table`, `procrustes_compare`, `protest`,
`align_to_reference`, `point_dispersion`) gained an optional `key=` argument
that applies such a normalisation before intersecting. Stored results are never
modified.

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
