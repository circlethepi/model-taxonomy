# Changelog

<!-- When opening a PR, always add an entry under "Unreleased" describing what changed and why. -->

## Unreleased

### Training length is settable as a total sample budget, and is a selection axis

**`fine_tuning.total_train_samples`** sets how many samples a model sees in total,
independent of how many distinct samples its dataset holds. A 1000-sample dataset with
a 10000 budget trains for 10 epochs; a 10-sample dataset for 1000. Previously training
length was only `n_epochs`, so across an `n_samples` sweep dataset size and optimizer-step
count were confounded and nothing could be attributed to the data alone.

Three values: `null` (off — train for `n_epochs`, exactly as before), an integer, or
`"auto"`, which resolves to one epoch of the full source data behind the recipe — the
summed split length for a simple recipe, the summed full size of every class used for a
class-aware one. `"auto"` is often far larger than `n_samples` (~280k against 1000 for the
two-topic Yahoo mixes), which is why the feature is opt-in for now; `--dry-run` on
`finetune_lora.py` prints the resolved budget, step count, and implied epochs per pair
without loading a model.

The budget is expressed to `SFTTrainer` as `max_steps`, the only unit it takes, so the
realized count quantizes to the effective batch (16 by default) and the log line reports
what was actually trained.

**Budgeted adapters get a `_b{samples_seen}` directory suffix**, recording what the model
actually saw rather than what was asked for. Two runs differing only in
training length would otherwise collide on one directory and the second would be skipped
as already-trained. Epoch-mode adapters keep exactly the names they have, so nothing on
disk moves.

**`training.n_epochs` now records passes actually made, not the configured value** — under
a budget `max_steps` overrides `num_train_epochs`, so a run configured for 3 epochs that
made 10 recorded 3. The request survives as `n_epochs_configured`, and `_build_entry`
derives the count from `samples_seen / n_samples` rather than trusting the field, which
reproduces the stored value exactly in epoch mode and corrects it under a budget.

**`CacheEntry` gained `samples_seen` and `n_epochs`**, so `index.filter(samples_seen=…)`
and `index.slices(by=("samples_seen", "seed"))` work at every taxonomy level — there is
one `CacheIndex`, narrowed per level by `with_available(...)`. Adapters trained before
this change fall back to `n_samples * n_epochs`, which is the same quantity, so the axis
is populated across the whole existing cache without a retrain or a migration.

### Behavioral generation samples, and stores several continuations per query

**`replicates`.** `BehavioralTaxonomy` draws `R` continuations per query via
`num_return_sequences`, and stores them all: the matrix is `(n_queries * R, d)`
in **query-major** order (`q0r0, q0r1, …, q1r0, …`) and `generated_texts` is
nested per query. Reading with `replicate_reduction="mean"` averages each query's
replicates back to `(n_queries, d)`, through the existing surrogate mechanism so
the reduction is computed once. The cache keeps the superset for the same reason
TODO item 12 settled query pooling on the read side: the mean is recoverable from
the rows, the spread is not recoverable from the mean. At 64 queries and `R=8`
an entry is 1.57 MB, so `05_generated` goes from 2.4 MB to ~21 MB — against
669 MB already in `04_activations`.

**Decoding now samples** (`do_sample=True`, with `temperature`, `top_p`,
`top_k`, `generation_seed`). Replicates are meaningless without it — greedy
decoding would store `R` copies of one continuation — so `replicates > 1` with
`do_sample=False` is rejected at construction rather than silently duplicating.

**Filenames grew two components**, because both change the numbers and neither
was anywhere else in the path:

    before: generations/generation128.json
            embeddings/generation128_{embedder}.safetensors
    after:  generations/generation128_8r_{sampling8}.json
            embeddings/generation128_8r_{sampling8}_{embedder}.safetensors

`save()` is idempotent *on the filename*, so without them a second run at a
different temperature would have found the first run's file, returned early, and
handed back the first run's numbers — the hazard already documented for
`torch_dtype`, on an axis that changes the result far more. `behavioral_selector`
gains `replicates`, `sampling_hash` and `replicate_reduction` to match, and
`_behavioral_variant_choice` resolves a 4-tuple parsed through `_GEN_RE` instead
of slicing the stem.

`mode_token` is untouched: it is the same function object on `ActivationCache`,
and widening it would move `04_activations` filenames for a parameter the
functional level does not have. The new components come from
`GeneratedTextCache.variant_token()`.

**Reproducibility is now conditional on `batch_size`.** One RNG stream serves a
whole `generate` call, so batch shape determines which tokens are drawn; the
per-batch seeding in `_seed_for_batch` makes a re-run at the same `batch_size`
and `generation_seed` exact, and a different `batch_size` genuinely different.
`batch_size` stays out of the cache key and stays in `metadata` and `runs/`.
The GPU check changed accordingly: `t_behavioral_batch_invariance` asserted a
greedy tie-flipping signature that sampling makes meaningless, and now asserts
same-seed reproducibility, that reseeding changes the text, and that replicates
within a query differ — reporting cross-batch divergence rather than failing on
it.

**Migrated by `scripts/migrate_behavioral_replicates.py`** (`--dry-run` /
`--apply` / `--revert`): 10 entries renamed to `1r_6f000f01` — the frozen greedy
sampling hash — tensors byte-identical, 640 generations nested, none dropped.
Apply-then-revert was verified byte-identical against the original tree. The
greedy entries are kept rather than superseded: they are what the 2026-08-05
measurement was computed from, and they are not comparable with sampled runs,
which is exactly why the hash is in the name.

### Recipes can compose several columns into their text — half of item 11

**`text_fields` and `text_separator`** on `ClassDatasetEntry` / `DatasetEntry`.
An entry may name several columns instead of one, joined by the separator, so
`question_title` + `best_answer` pairs are expressible. `finetune_lora.py`
synthesizes a `_composed_text` column and hands *that* to `SFTTrainer`, since
`dataset_text_field` takes one column name.

This exists because the 25 yahoo adapters were fit on bare `best_answer` prose —
no question, no template — while behavioral extraction prompts them with a
`question_title` to continue. The generations therefore came from an input shape
that never appeared in training, and behavioral recovered the mixing order
**exactly backwards** (r = −0.9995, Procrustes p = 0.955).

**The addition is additive, not a migration.** `recipe_hash` is a SHA-256 over
`to_dict()` output, so emitting the new keys unconditionally would have changed
all six existing recipe hashes at once and orphaned 523 draws, 25 adapters and
everything keyed on them. `composition_dict()` returns nothing when no
composition is set, so an entry that does not use it serializes byte-identically;
a check pins the six known hashes. The three copies of the row→text loop in
`mixed_dataset.py` collapsed into `_text_projection.row_text`.

`experiments/yahoo_qa_model_train.yaml` retrains the five mixtures on pairs, with
every LoRA parameter read off the adapters on disk so the projection is the only
difference, under `yahoo_qa_*` names so nothing is overwritten. The query recipe
is unchanged and still hashes to `04a65e58df502e45`, so old and new adapters land
at the same draw coordinates. `02_dataset_embeddings` deliberately still embeds
`best_answer` — moving it re-embeds 520 entries and is a separate decision, so a
cross-taxonomy comparison against these adapters scores a QA-trained model
against a `best_answer`-derived dataset geometry.

### Every stage now spells a draw the same way, and `02` finally says which one

**`src/cache/_draw.py` owns the draw token.** `n{n}_s{seed:02d}`, zero-padded.
Three stages had three spellings for one coordinate: `01_datasets` wrote
`n100000_s0.json`, `04`/`05` wrote `n64_s00`, and `02_dataset_embeddings` wrote
no draw at all. Nothing detected the drift because each stage only read its own
names — `DrawKeyedCache.draw_name` even documented its token as matching
`01_datasets` while `01` wrote something else. Writing is narrow and reading is
wide: `draw_name` always pads, `parse_draw_name` accepts both widths, so
pre-migration names still read.

**`02_dataset_embeddings` gains a draw level and a surrogate level**, matching
`04`/`05`:

    before: {recipe_hash}/{embedder_hash}/{config.json, embeddings.safetensors}
    after:  {recipe_hash}/n{n}_s{seed:02d}/{embedder_hash}/
                config.json
                surrogates/{surrogate_hash}/{config.json, surrogate.safetensors}

`embedder_hash` now keys `{embedder_config}` alone and `representation` moved
into the surrogate spec. Folding the draw into that hash was a real fix once —
when `recipe_hash` became content-addressed, something had to stop two seeds of
one mixture sharing an entry — but it left `02` the only stage where "which draws
are embedded?" needed one JSON parse per entry. **The guarantee moved into the
path; it did not go away.** All 520 stored entries had a constant
`representation`, `model_name` and `prompt_prefix`, so the directory was in
practice a draw hash.

**`dataset_selector`** joins `behavioral_selector` and `functional_selector`,
since `representation` left the hash and `embedder_hash=` alone can no longer
express which representation to read. It is one more axis
`CollectionCache.collection_hash()` cannot see — that is TODO item 14.

**`_embedder_choice` collapsed from ~50 lines to a set intersection.** It only
did signature bookkeeping because the hash bundled `n_samples`, which meant one
shared hash implied one shared sample count and ruled out the pooled comparison
across the size sweep.

**A `02` surrogate is authored, not derived**, and this differs from `04`/`05`.
Those store a base artifact and compute views from it at read time; `02` stores
no base, because the full `(N, 768)` embeddings would cost 6.1 GB and a GPU
re-embed of ~2M texts, and `mean` is not invertible. Adding a representation here
means re-embedding. Recorded in the class docstring.

**Migration** (`scripts/migrate_dataset_embedding_layout.py`, additive with a
separate `--prune`): 523 draws renamed and 520 entries relayed out, all verified
byte-identical then pruned. No GPU, no recomputation, no distance changed.
`verify_sampled_cache --full` rehydrated all 1046 draws (both spellings) against
their recorded checksums between apply and prune: 0 failed.

**Also fixed:** `scan_yahoo_cache` / `scan_yahoo_cache_detailed` reported **1
draw per proportion instead of 94–135**, and dropped one proportion entirely.
They regexed the recipe *name*, which content-addressing had reduced to one
arbitrary stale label per directory; they now list draw directories and derive
the proportion from `normalized_class_weights`. Two stale docstrings in
`DatasetEmbeddingTaxonomy` were corrected — `representation` has three modes, not
two, and `gram` no longer mirrors `FunctionalTaxonomy`.

`scripts/migrate_recipe_identity.py` now computes the old four-field embedder
digest itself. A completed migration is a record of a transition that already
happened; it must not change meaning when the live signature does.

`check_analysis.py`: 60 passed, 2 failed, 1 skipped across 63 checks. Both
failures pre-date this work — `migrate_behavioral_layout.py --apply` has never
been run against the shared cache, so the old run-wise `05_generated` directories
survive.

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
