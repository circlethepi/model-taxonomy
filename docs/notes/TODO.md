# TODO — outstanding work, and where each piece is specced

Every note in this directory records work that was agreed and then deliberately
deferred, together with the decisions already taken and the call sites already
verified. This file is the index: what is outstanding, and which note holds the
detail.

**Read the linked note before starting an item.** Each one already contains the
verified call-site list and the reasoning behind the decisions, so none of it needs
re-deriving. Two caveats that apply to all of them: notes are point-in-time, so
confirm the cited paths still exist first; and line numbers move, so grep rather
than trusting them.

The numbering is for reference in conversation. It is not a strict execution order.
Dependencies are called out in the State column.

## Items

| # | Item | Note | State |
|---|---|---|---|
| ~~1~~ | ~~Number the shared cache to `01_datasets` … `06_collections`~~ | [cache_layout_migration.md](cache_layout_migration.md) | **Done.** Migrated by `scripts/migrate_cache_layout.py`; recipes now have a hash-indexed home at `01_datasets/{recipe_hash}/recipe.json`, and the structural `DiskCache` fallback was removed. |
| ~~2~~ | ~~Sampled-dataset storage, and whether n/seed belong in recipe identity~~ | [sampled_dataset_storage.md](sampled_dataset_storage.md) | **Done.** `recipe_hash` is now content-addressed over `{recipe_type, datasets}` — 880 old hashes collapsed to 6 — and draws are stored as source indices (2.07 GiB → 40 MiB), keyed `n{n}_s{seed}.json`. Migrated by `scripts/migrate_recipe_identity.py`. The row-ordering/prefix-nesting scheme in the note is now moot: indices already beat it. |
| 3 | `scripts/run_comparison.py` + a YAML `comparison:` block | [comparison_followups.md](comparison_followups.md) §1 | One decision open: build a `CacheIndex` from resolved model IDs, or let `comparison.models` accept the same `base_models` / `fine_tuned` tokens the other config sections use. The second is more consistent. |
| 4 | `src/plots/simplex.py` + `notebooks/4_taxonomy_comparison.ipynb` | [comparison_followups.md](comparison_followups.md) §2 | Five figures specced: ternary/simplex, recovered-vs-true, cross-taxonomy disparity heatmap, Shepard diagram, convergence-in-`n`. Load the `dataviz` skill first, and match `src/plots/config.py`. |
| 5 | `docs/guides/pipeline_stages.md` — a runnable template YAML per stage | [comparison_followups.md](comparison_followups.md) §3 | Unblocked by 1 — the "what it writes to the cache" column can now quote the final directory names. |
| 6 | Add distance correlation (dCor); demote Mantel's p-value to descriptive | [distance_matrix_comparison.md](distance_matrix_comparison.md) §1 | Mantel has inflated type-I error on dependent off-diagonal entries; PROTEST is already implemented and better calibrated. Do **not** drop Mantel — existing figures reference it. |
| 7 | Fit `T: D_taxonomy → D_simplex` and test whether it converges in `n` | [distance_matrix_comparison.md](distance_matrix_comparison.md) §2 | Research, not plumbing. The data already exists: the `by_seed/` grouping varies `n_samples` at fixed seed, and `compare_taxonomies` already persists a restricted `T_n`. Fit on distance matrices, not raw MDS coordinates. |
| 8 | Make `src/metrics/` handle variable-length representation rows | in-code TODOs at `src/metrics/cka.py:1`, `src/metrics/frobenius.py:1`, `src/taxonomy/structural.py` | Blocks the full structural pipeline through `ModelRepresentation`. Storage format still undecided (zero-pad / per-module-type / a ragged type). Also relevant to 9 — see its open question 4. |
| ~~9~~ | ~~Functional/behavioral: run them, then wire them into the comparison layer~~ | [functional_behavioral.md](functional_behavioral.md) | **Both done.** Behavioral: `GeneratedTextCache`, `behavioral_repr`, `_behavioral_matrix`, checks at all three tiers. Functional: `ActivationCache` (model-wise, per-layer, additive), `functional_repr`, `_functional_matrix` + `functional_selector`, a shared `HFInferenceTaxonomy` base class, and checks at all three tiers. Open questions 1–3 are closed (see the note); **question 4 remains** — routing through `src/metrics/` for variable-length rows, which is item 8. `gram` was **redefined**: rows are queries, not layers — see [gram_and_cka.md](gram_and_cka.md). |
| 11 | **`text_field` disagrees between training and behavioral extraction** | this row; see also 2, 12 and [embedder_task_prefixes.md](embedder_task_prefixes.md) | **Half done — the mechanism exists and five adapters are queued to be retrained; the dataset-level half is still open.** Training sets `text_field: best_answer` and `finetune_lora.py` hands it to `SFTTrainer` as `dataset_text_field`, so adapters are fit as a plain causal LM on **bare answer prose — no question, no template**. The behavioral query set sets `question_title`, so extraction prompts them with a **question** to continue. That shape never appeared in training, so behavioral distances are measured out-of-distribution. Same signature as the nomic prefix bug: no error, fluent output, effect invisible until measured against a correctly-formed input. **Not a one-line fix** — `ClassDatasetEntry.text_field` is a single string selecting one column at read time, so question+answer is not expressible in YAML and needs a *composition* mechanism in the recipe layer. **Blast radius:** it changes `recipe_hash`, so every affected draw re-materialises and all 25 adapters need retraining. **It splits the levels:** dataset embeddings also use `best_answer`, so if training moves to combined text and the dataset level does not follow, "what a 75/25 mixture means" stops being the same thing at each level — decide both together. **It revives item 2's parked question:** if `text_field` becomes a composition rather than a column selector, the "it is only a read-time projection" argument for keeping it out of the recipe hash weakens considerably. | **What was built:** `ClassDatasetEntry`/`DatasetEntry` gained `text_fields` + `text_separator` (`src/datasets/_text_projection.py`), `finetune_lora.py` synthesises a `_composed_text` column for `SFTTrainer`, and `experiments/yahoo_qa_model_train.yaml` retrains the five mixtures on `question_title` + `best_answer` joined by a newline, with every LoRA parameter read off the adapters on disk and `yahoo_qa_*` names so nothing is overwritten. **The blast radius did not materialise.** This row predicted every affected draw would re-materialise; in fact the *composed* recipes get new hashes (`b5c09a54…`, `c54ccf70…`, `29a863b8…`, `bedc8bc0…`, `c115f05f…`) and the existing six are byte-identical, because `composition_dict()` omits the keys when unset and `recipe_hash` is a hash of `to_dict()` output. Additive, not a migration. **Still open, and it is the half this row cares most about:** the dataset level. `02_dataset_embeddings` still projects through `best_answer`, so "what a 75/25 mixture means" is now genuinely two different things at the training and dataset levels. Moving it re-embeds 520 entries. Until then, any cross-taxonomy score involving a `yahoo_qa_*` adapter is comparing a QA-trained model against a `best_answer`-derived dataset geometry, and that belongs in the caption. **Also still open:** the re-measurement itself — the adapters are not trained yet.
| 16 | **Behavioral stores one sample per query** | this row; see also 11, 12, 14 | **Done (code); the run is pending.** `BehavioralTaxonomy` samples (`do_sample=True`, temperature/top_p/top_k/`generation_seed`) and draws `replicates` continuations per query via `num_return_sequences`. Stored `(n_queries * R, d)` **query-major**; `replicate_reduction="all"\|"mean"` chooses at read time, through the surrogate mechanism, so the cache stays the superset (the item-12 principle: the mean is recoverable from the rows, the spread is not recoverable from the mean). Filenames gained `_{R}r_{sampling8}` — **not decoration**: `save()` is idempotent on the filename, so a second run at another temperature would otherwise return the first run's numbers silently. `mode_token` was left alone so `04_activations` filenames did not move. **The reproducibility guarantee changed shape:** one RNG stream serves a `generate` call, so `batch_size` now determines the text (first-order, where greedy's effect was last-bit); same batch + same seed is exact, different batch is not, and `batch_size` stays out of the cache key but in `metadata`/`runs/`. The GPU batch-invariance check was rewritten around that. Migrated by `scripts/migrate_behavioral_replicates.py`: 10 entries → `1r_6f000f01`, tensors byte-identical, 640 generations nested, apply-then-revert verified byte-identical. Greedy entries kept, not superseded — they are what the 2026-08-05 table below was computed from. **Pending:** the R=8 extraction over both adapter sets and the four-cell re-measurement.
| 10 | Generalize CKA to multi-layer / multi-projection sweeps | [cka_notes.md](cka_notes.md) | Parked on a genuine modeling choice, not on effort. Needs a decision in conversation before any code. The other three distance builders were already generalized — see [frobenius_bw_generalization.md](frobenius_bw_generalization.md) for those proofs. |
| 12 | **Behavioral representations have no `representation:` knob, unlike the dataset level** | this row; see also 8 | **What is on disk today** (verified against `05_generated/5191ad734b81daff/`): `embeddings/{slug}.safetensors` holds `matrix` as **`(n_queries, 768)` float32** — one row per query, each row L2-normalized, no pooling across queries — plus a `_meta_json` blob carrying `model_id`, `taxonomy` and `metadata`. The generated **text is not in the tensor file**: it lives beside it in `generations/{slug}.json`, and `GeneratedTextCache.load()` folds it back into `metadata["generated_texts"]` so callers see the original object. At `n_queries=64` that is ~192 KB of tensor + ~33 KB of text per model, growing linearly in `n_queries`. **The asymmetry:** `DatasetEmbeddingTaxonomy` takes `representation: mean\|matrix` and the populated `02_dataset_embeddings` all use `mean` (a `(1, d)` centroid), but `BehavioralTaxonomy` has no such parameter — it always stores the full matrix. **Why that matters less than it looks:** `CosineDistanceMetric` flattens both matrices before comparing, and because row *i* is query *i* for every model and every row is unit-norm, the flattened cosine reduces exactly to the **mean per-query cosine similarity** — `cos(A_flat, B_flat) = (1/n)·Σᵢ⟨aᵢ,bᵢ⟩`. So the metric already averages over queries; keeping the full matrix costs disk but is what makes per-query analysis (centering, subsetting, per-query variance) possible at all — and those are the levers if the signal stays weak. **Decide:** whether behavioral gains a `representation:` option for symmetry, and whether a pooled `(1, 768)` variant is stored alongside so behavioral and dataset centroids are directly comparable. Note a pooled behavioral vector is **not** the same as the flattened cosine — averaging vectors then comparing differs from comparing then averaging. **Partially addressed on the read side:** `build_taxonomy_artifacts(..., behavioral_selector={"representation": "matrix"\|"mean", "renormalize": bool})` now pools at read time (`_behavioral_view` in `src/analysis/comparison.py`), mirroring `functional_selector`. These two keys live in the *same* selector as the item-13 addressing keys (`draw`, `max_new_tokens`, `embedder_hash`, `view`, `normalize`) and are applied after them. The pooling knob is spelled `renormalize`, **not** `normalize`, because the latter was taken by the cache's own row-wise mode (`"layer"\|"global"\|"none"`, folded into the surrogate hash and applied *before* pooling); the two compose and are not alternatives. `BehavioralTaxonomy` itself is untouched and still stores only the full matrix — which is the right side to leave alone, since pooling is lossy and the cache should stay the superset. `renormalize` is a separate key because the stored rows are unit-norm, so their mean is not (measured 0.712–0.724 over the five yahoo adapters); un-normalized, `1 - dot(mean, mean)` is 0.482–0.492, i.e. a constant offset with only ~0.010 of real spread. **What is still open:** whether the *extraction* side gains the knob, and whether a pooled variant is stored alongside. |

| ~~13~~ | ~~The two inference caches diverge on three axes~~ | this row; see also 9, 14 | **Done — all three axes, and (b) was worse than this row claimed.** **(a)** unified: `05_generated` is now model-wise, sharing `{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/` with `04_activations` via a new `src/cache/_draw_keyed.py::DrawKeyedCache`. The row said this was "not obviously worth unifying"; it was, because fixing (b) *is* a migration of the model key and doing it twice would have cost the same twice. **(b)** fixed. The row said moving the cache root would orphan the entries — in fact **they were already orphaned**: the stored `model_id`s were *relative* paths, so `behavioral_repr` resolved to **0 hits across all 25 adapters × 2 configs** while 10 representations sat readable on disk. It went unnoticed because both checks touching behavioral bypassed the broken path — one read by config hash, the other excluded behavioral unless exactly one config existed, and there were two — so the level had **never** been exercised end to end. It now is, and `behavioral_repr` resolves to 5 from any cwd and from a relative root. **(c)** kept **as this row wrote it**: no query text is stored. A draft of the fix proposed inverting it, on the grounds that `text_field` is a read-time projection outside `recipe_hash`. That is false — `ClassDatasetEntry.to_dict()` includes `text_field`, `_canonical()` hashes the entries, so `(recipe_hash, n_samples, seed)` *does* determine the text. `_replay_queries` guessing the column was a bug in `_replay_queries`, now reading `text_field` from `recipe.json`. **Do not re-propose storing query text.** Also: `model_slug` is gone (its one live use was an in-memory PEFT adapter name, now `_hf_inference._adapter_name`), `source_indices` is populated rather than always empty, and `compare_all_slices` finally forwards the selectors it never had. Migrated by `scripts/migrate_behavioral_layout.py` (`--dry-run`/`--apply`/`--revert`): 10 entries byte-identical, 5 sharing coordinates with `04_activations`, 128 stored query strings dropped as redundant. |

| ~~14~~ | ~~`CollectionCache.collection_hash()` ignores the taxonomy selector~~ | this row; see also 9, 13 | **Done.** The key is now `{taxonomy}/{collection_key}/{metric}_{surrogate_key}`, built from **what each model actually resolved to** rather than from the caller's selector: `collection_key` hashes the ordered `(model_id, artifact_path)` pairs — the cache-root-relative path, stopping *before* `surrogates/` — and `surrogate_key` hashes the ordered list of per-model surrogate hashes. Selectors are not hashed directly, deliberately: `{}` and `{'draw': ...}` are the same read but different dicts, and `{}` changes meaning once a second draw exists. Every cache's `load()` now surfaces `artifact_path` and `surrogate_hash` in the representation metadata, and `build_taxonomy_artifacts` **resolves before it keys** — the representations are loaded either way, and the cache exists to skip the pairwise computation, which is the expensive part. **The predicted cost did not materialise:** "invalidates every stored collection" was 4 entries and 52 KB, rebuildable on CPU. **Confirmed against the numbers this row recorded:** the quarantined functional entry rebuilds to **4.088e-10** under `normalize="global"` and **1.126e-03** under `layer` — matching 4.09e-10 / 1.13e-03 exactly, so the stored bytes were intact and the blind spot was the whole discrepancy. Pinned by `t_collection_key_sees_selector`, which asserts two normalizations give two directories under **one** `collection_key` and that their matrices differ. **Surrogate hashes are digested as a list, not asserted equal.** They coincide today — 13 of 13 functional surrogates are present for all 5 adapters, because a surrogate spec carries the shared *query* draw's recipe hash, not the model's training recipe — but that is a property of the data, and it breaks as soon as models are extracted against different query datasets. **Also fixed here:** the metric-name mismatch (lookup used `'cka'`, save used `'cka_linear'`, so such a collection never hit and rewrote its directory every run — `_resolve_metric` now accepts both spellings); `_fit_geometries` ignoring `mds_kwargs`, so a new `random_state` returned the old coordinates; `_functional_matrix` silently ignoring misspelled selector keys; and `cache_root=None` not disabling the cache, now a separate `use_cache=False`. Quarantined by `scripts/migrate_collection_key.py` (`--apply`/`--compare`/`--prune`/`--revert`): the functional entry was pruned once its rebuild reproduced it; the **three behavioral entries are still quarantined** in `06_collections/_legacy/`, since their `slice` records pre-item-13 vocabulary and their inputs were being rewritten by the item-16 re-extraction. Rebuild and prune them once that has settled. |

| ~~15~~ | ~~`02_dataset_embeddings` hides the draw in a hash~~ | [dataset_embedding_layout.md](dataset_embedding_layout.md) | **Done.** `02` is now `{recipe_hash}/n{n}_s{seed:02d}/{embedder_hash}/surrogates/{hash}/`, matching `04`/`05`; `embedder_hash` keys `embedder_config` alone and `representation` moved into the surrogate spec. **The draw token was unified across three stages, not two** — `01` wrote an unpadded seed while `04`/`05` padded theirs, so `src/cache/_draw.py` now owns the spelling and `01`'s 523 draws were renamed to match. Migration was content-preserving: 523 + 520 verified byte-identical, then pruned; no GPU, no distance changed. Also fixed in passing: `scan_yahoo_cache` reported 1 draw per proportion instead of 94–135 and dropped one proportion entirely. Adds `dataset_selector`, which is one more axis item 14 cannot see.

## First end-to-end measurement of the behavioral level (2026-08-05)

Item 13 made this possible: behavioral had **never** joined a cross-taxonomy
comparison, because the check that would have run it required exactly one cached
config and there were two. It now joins, and the first numbers are bad in a
specific and interpretable way.

Same slice as the functional smoke run — 5 adapters, `n_samples=1000, seed=0`,
64 queries, draw `04a65e58df502e45/n64_s00`, cosine:

| taxonomy | stress | r | ρ | Procrustes | p |
|---|---|---|---|---|---|
| dataset_embedding | 0.2879 | 1.0000 | 1.0000 | 0.0041 | 0.005 |
| functional | 0.0193 | 1.0000 | 1.0000 | 0.2487 | 0.015 |
| structural | 0.0534 | 0.9937 | 1.0000 | 0.4500 | 0.015 |
| **behavioral** | 0.2156 | **−0.9995** | **−1.0000** | **0.9772** | **0.955** |

**Behavioral recovers the mixing order exactly backwards**, and its Procrustes
p-value is 0.955 — no better than chance. The other three levels are all
positively ordered and significant.

Do not read this as "behavioral does not work" yet. It is exactly the failure
**item 11** predicts: the adapters were trained on bare `best_answer` prose with
no question and no template, while the behavioral query set prompts them with a
`question_title` to continue. The generations are therefore produced from an
input shape that never appeared in training. A monotone-but-inverted ordering is
more consistent with a systematic artefact than with noise, which would scatter.

**Before treating this as a property of the level, fix item 11 and re-measure.**
The re-measurement is cheap now that the level joins a comparison at all.

## Small wins

Recorded so they survive; each is cheap and independent.

- ~~**Drop torch from the distance-computation import path.**~~ Done. Verified all
  5,600 tensors across all 25 adapters are `F32` (read from the safetensors headers,
  no tensor data), then switched `src/notebook/lora_weights.py` to
  `safe_open(framework="numpy")`. Distances are bit-identical. A `bfloat16` adapter
  would need `framework="pt"` and a `.float()` before `.numpy()`.
- ~~**Make `src/__init__.py` lazy.**~~ Done, and this was the real cost: it eagerly
  imported transformers, sentence-transformers, torch and umap/numba, so *any*
  `import src.x` paid ~470 MB and ~7 s before doing any work. Now PEP 562
  `__getattr__` over a name → module map. `scripts/check_analysis.py` went
  **587 MB / hung the IDE → 222 MB / 6.7 s** for the full 35 checks; the
  cache-and-discovery path alone is 52 MB and never imports torch.
- ~~**Fix the nonexistent default in `load_lora_weights`.**~~ Done during 1 — it
  defaulted to `results/shared_cache/peft_adapters`, which never existed; now
  `results/shared_cache/03_adapters`.
- ~~**Read the configured device for the sentence embedders.**~~ Done. Both
  `SentenceTransformerEmbedder` construction sites hardcoded `device="cpu"`, so every
  sentence embedding the pipeline had produced — including the whole populated
  `02_dataset_embeddings` cache — was computed on CPU. Now routed through
  `_utils.resolve_device`: an explicit per-embedder `device`, else
  `extraction.device`, else auto-detect. `device` is deliberately outside
  `config_dict()`, so no cache key changed and nothing was invalidated.
- ~~**Prepend nomic's task-instruction prefix.**~~ Done. `prompt_name="document"` was
  a **silent no-op** — nomic ships no `prompts` map, sentence-transformers synthesises
  `{"query": "", "document": ""}`, and the empty string was prepended without error.
  All 520 cached dataset embeddings are therefore bare-text. They are **geometrically
  equivalent** (MDS recovery 0.9977 vs 0.9976 pearson, spearman 1.0000 both ways) and
  are being kept, but they are unreachable under the new `embedder_hash` by design.
  Full write-up, including the `max_seq_length` cap to use if you ever batch a
  re-embed: **`docs/notes/embedder_task_prefixes.md`**.

## Do not "simplify" these

- **`compare_taxonomies` projects from the largest available dimension, not `k-1`.**
  Projecting from exactly `k-1` makes the anchors' affine hull the whole space, every
  residual becomes identically zero, and the off-simplex diagnostic silently
  disappears. Pinned by `t_projection_dimension_matters` in
  `scripts/check_analysis.py`.
- **`src/core/analysis.py` uses `path / "representations"`, and that is not the
  shared cache.** It is the layout *inside* a saved `TaxonomyAnalysis` directory.
  Renaming it would break `ModelTaxonomyProfile.load` on every existing saved
  profile. Left untouched by 1 on purpose — see the "Two traps" section of
  [cache_layout_migration.md](cache_layout_migration.md).
- **Converting a stored draw to indices must content-match, not re-sample.** The
  obvious implementation — re-run the sampler and record what it selects — answers
  "what would we draw now", not "what was drawn", and those differ for any draw the
  sampler no longer reproduces. `source_registry.locate_rows` matches stored rows back
  to source positions instead.

  5 draws were affected, all in `yahoo_075t0_025t1`, whose capacity is 186,666:
  n=190000 seeds 0–2 (187,500 rows stored vs 186,666 today) and n=265000 seeds 0–1
  (206,250 vs 186,666). They predated the proportional class scale-down in
  `ClassMixedDataset._load_entry`, which caps the *total* to preserve class ratios where
  the older code let per-class counts clip individually.

  **Those 5 have since been deleted, along with the 4 dataset embeddings derived from
  them**, so the cache now contains nothing the current sampler would not reproduce
  (521 draws, verified). Both deleted n values exceeded the mixture's capacity anyway —
  under today's semantics a request for either clamps to the same 186,666-row draw, so
  they were never the distinct sweep points their names claimed. No adapter was trained
  on them. Nothing regenerates them: `n_samples_sweep: nice 4` tops out at 500,000 and
  `apply_dataset_capacity_caps` collapses everything above capacity to a single
  n=186666 block.

  The rule survives the cleanup: **any future migration of stored draws must recover
  indices by content-matching, not by re-running the sampler.** Sampling logic will keep
  evolving, and a stored draw is a record of what was drawn, not a recipe for redrawing.
- **`{output_dir}/adapters/` is deliberately unnumbered.** The numeric prefixes
  describe shared-cache pipeline stages; an experiment output directory has no such
  ordering, and renaming it would disturb the `LoRACache` slugs in the legacy
  per-experiment caches, which encode the path literally.

## Verification

`python scripts/check_analysis.py` is the sharpest instrument in the repo and the
only test harness — there is no pytest. It registers 63 checks (51 synthetic, 12 that
read the real cache); the data-backed ones report `SKIP` when a path is missing
rather than passing vacuously, so they fail loudly on a missed migration. Run it
before and after any of the items above and compare the pass/skip profile.

**A green suite is not proof a level works.** Item 13 is the cautionary case: the
behavioral cache was unreachable through discovery for its entire life while the
suite read 54/0/0, because the only two checks touching it both went around the
broken path. When adding a level, make sure at least one check exercises it the
way a *caller* would — through `scan_cache` and `build_taxonomy_artifacts` — not
just by reading its files back directly.

`python scripts/verify_sampled_cache.py --full` is the second instrument, and the one
that matters for the dataset cache specifically: draws are stored as source indices, so
it rehydrates each one and checks it against a recorded `rows_sha256`. Nothing else
would notice if the upstream HuggingFace dataset changed underneath. `--fast` checks
revisions and row counts only and takes seconds.

The full run is ~7 s and ~220 MB, so it is cheap enough to run freely. Four flags
narrow it further when you want a fast loop:

    --list              print the check names, run nothing
    -k PATTERN          only checks whose description contains PATTERN
    --data-only         only the 12 that read the real cache — what catches a
                        broken path after a migration
    --synthetic-only    skip the cache entirely
    --include-gpu       additionally run the [gpu] tier, which loads a real model
                        onto a CUDA device. Off by default; the SLURM job passes it
                        after extraction, while the GPU is still allocated.

Run it in the project conda environment (`conda activate taxonomy-env`, the same one
the SLURM scripts activate) — `numpy` is not installed in the base env, so the
checks fail at import there. Baseline as of 2026-08-07, after item 14:
**67 passed, 0 failed, 0 skipped** across 67 registered checks, plus 2 `[gpu]`
checks behind `--include-gpu`.

⚠️ **The previous entry here was stale in every number, and the stated cause was
wrong — re-measure rather than trusting this paragraph.** It read "60 passed, 2
failed, 1 skipped across 63" and blamed the 2 failures on
`scripts/migrate_behavioral_layout.py --apply` never having been run against the
shared cache. It *had* been run: `05_generated/` is fully model-wise with zero
old-style run-wise `{config_hash}/` directories. The real state at the start of
item 14 was **65 passed, 1 failed** across 66, and the one failure was new — the
item-16 re-extraction had landed a second behavioral variant (`8r`), so
`t_comparison_end_to_end` named a *draw* but not a *variant* and
`_behavioral_variant_choice` rightly refused to guess. Fixed by
`_shared_behavioral_variant`, which picks deterministically.

That is the item-13 lesson recurring for the third time: **a check that assumes
exactly one of something silently stops exercising the level when a second
appears.** When adding a check over an inference level, name the variant.

Both inference levels are
populated, so nothing skips; the `[data]` checks for each must **skip**, never
pass, whenever their stage directory is absent — verified by hiding `05_generated`,
which turns exactly 3 `[data]` checks to SKIP and fails none. Earlier baselines:
54/0/0 after layerwise normalization, 53/0/0 after the functional work, 45/0/1 after the
behavioral work (the one skip being functional's absent `04_activations/`), and
40/0/0 before either. `t_scan_cache` reports 25 adapters,
**25 with recipes, 25 usable**, and `verify_sampled_cache --fast` reports **523** draws
ok, 0 legacy, 0 failed. (The "521" recorded here previously was already stale before
the functional work — the count was 523 both before and after that run, which reused
an existing draw rather than adding one. Re-read the number; do not trust this line.)

### What the functional smoke run showed (job 1999300, L40S, 1m48s)

Five adapters spanning the mixing range, 64 queries, all 29 hidden states, `input`
mode, mean pooling — `experiments/yahoo_functional_smoke.yaml`. Stored `(64, 3072)`
per layer per model; the `concat` view is `(64, 89088)`.

**The level carries signal, and the ordering is exactly right.** CKA distance from
the pure topic-0 anchor is **monotone** in the true mixing proportion:

| adapter | true topic-0 | distance from 100/0 |
|---|---|---|
| `yahoo_100t0_000t1` | 1.00 | 0.0000 |
| `yahoo_075t0_025t1` | 0.75 | 0.0045 |
| `yahoo_050t0_050t1` | 0.50 | 0.0084 |
| `yahoo_025t0_075t1` | 0.25 | 0.0098 |
| `yahoo_000t0_100t1` | 0.00 | 0.0107 |

Against the full pairwise `|mixing gap|`: **Pearson r = 0.803** (p = 0.005),
**Spearman ρ = 0.833** (p = 0.003). Off-diagonal spread 0.0011–0.0107, non-constant
and finite. In the end-to-end slice: stress 0.0212, r = 1.0000, ρ = 1.0000,
Procrustes 0.2399 (p = 0.015) — better than structural (0.4500) and worse than
dataset_embedding (0.0041). Functional↔behavioral correlate at **0.382**;
functional↔structural at **−0.006**.

**These numbers are all `normalize="global"`**, which was the only mode when the
job ran. See the re-measurement below.

### Re-measured under layerwise normalization (no GPU; read-time rebuild)

`normalize="layer"` is now the default. Same stored activations, same draw, same
metric — only the surrogate was rebuilt.

| | `global` (as run) | `layer` (new default) |
|---|---|---|
| off-diagonal min / max | 0.00113 / 0.01068 | 0.00110 / 0.01120 |
| spread | 0.00955 | 0.01010 |
| monotone in mixing proportion | yes | yes |
| Pearson r vs mixing gap | 0.803 (p = 0.005) | **0.823** (p = 0.004) |
| Spearman ρ vs mixing gap | 0.833 (p = 0.003) | **0.788** (p = 0.007) |

The two matrices correlate at **r = 0.993**. Layerwise separates the adapters
very slightly more and moves Pearson up and Spearman down — one rank swap. On
this evidence the choice of normalization is **not** what is limiting the signal;
the tiny absolute distances above still are.

**The premise behind layerwise turned out to be only half right, so record what
was actually measured.** The expectation was that deep layers dominate a
`global`-normalized concatenation. On Llama-3.2-3B, mean-pooled over these 64
queries, they do not: transformer-block row norms run 56 → 71, within ~1.6× of
each other, and the effective number of contributing layers is 26.4 of 29. What
*is* lopsided is the bottom: the **embedding layer and layer 1 have row norms
0.36 and 2.41**, giving them 0.00% and 0.01% of a row's squared norm. Under
`global` those two are effectively absent from a comparison labelled "all
layers"; under `layer` they count as much as any other. That — not depth — is
what the new default changes here, and it cuts both ways, since it also amplifies
a near-zero-norm layer and whatever noise it carries up to parity.

✅ **`CollectionCache` now sees the selector — this caveat is closed (item 14).**
It read: `collection_hash()` keys on `(model_ids, taxonomy, metric)`, so the
draw, layers, view and normalization are not in it, and a collection built before
the normalization change keeps returning its `global` matrix whatever selector is
passed. The figures above were therefore produced by going through
`_compute_distance_matrix` directly, which is still the right call for a sweep
you do not want cached (or pass `use_cache=False`).

The key is now `{taxonomy}/{collection_key}/{metric}_{surrogate_key}`, so the two
normalizations are two directories. The `global`/`layer` pair in the table above
was re-derived through the new key as part of that work and reproduces to
**4.088e-10** and **1.126e-03** respectively — i.e. the numbers in this section
stand, and the stored entry they came from was intact all along.

**The caveat worth carrying forward: the absolute distances are tiny.** A CKA
distance of 0.011 is a CKA *similarity* of 0.989 — all five models are nearly
identical in activation space, which is what you would expect from rank-16 LoRAs
over one frozen base. So the ordering is recovered from a **small perturbation on a
dominant shared base-model geometry**, not from a well-separated set of points.
That is a real result but a fragile-looking one: it is worth checking whether the
separation survives more adapters, more seeds, and a larger query draw before
treating functional distances as interchangeable with the other levels'.

**Batch invariance, measured rather than assumed** (`[gpu]` check, same job): batch 1
versus one batch of 8 gives **min per-row cosine 0.999999**, max|Δ| 7.81e-03. That is
fp16 matmul tiling and nothing else — the padding contribution the note asked to
quantify is, after mask-aware pooling, zero to six decimal places.

**Disk cost was ~2× the estimate**, and the reason is worth knowing: 219 MB for five
models, not the ~114 MB the plan projected. The projection covered the stored
activations (29 × 64 × 3072 × 4 B ≈ 22.8 MB per model) but not the **written-back
surrogate**, which is the same size again because the default `concat` view over all
layers is exactly as large as the layers it concatenates. Still cheap, but a
surrogate is not free the way a small derived view would be.

**And each normalization mode costs another full-size copy.** After the layerwise
change, `04_activations/` holds three surrogates per model — the orphaned
`{"normalize": true}` one from the smoke run, plus `global` and `layer` from the
re-measurement — 327 MB in total against 114 MB of actual activations. **109 MB of
that is the orphan**, unreachable because `true` now canonicalizes to `"layer"`
and hashes differently. It was deliberately **not** deleted: dropping cache entries
as a side effect of a code change is the wrong default. To reclaim it:

```bash
# inspect first, then remove
grep -l '"normalize": true' results/shared_cache/04_activations/*/*/*/*/surrogates/*/config.json \
  | xargs -n1 dirname
```

The general lesson for `n_queries=64`: a `concat` surrogate over all layers is
~23 MB per model per mode, so surrogates dominate this stage's footprint and will
keep growing one copy at a time as views and normalizations are explored.

The jump from 20 usable to 25 is task 2's doing, not a change in the data: the five
oldest adapters were trained under the pre-rename naming (`yahoo_25t0_75t1`) and used to
dangle, because their recipe hash matched nothing in the shared cache. Content-addressing
merged them with their zero-padded twins, so they now resolve.
