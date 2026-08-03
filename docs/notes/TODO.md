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
| 11 | **`text_field` disagrees between training and behavioral extraction** | this row; see also 2 and [embedder_task_prefixes.md](embedder_task_prefixes.md) | Training sets `text_field: best_answer` and `finetune_lora.py` hands it to `SFTTrainer` as `dataset_text_field`, so adapters are fit as a plain causal LM on **bare answer prose — no question, no template**. The behavioral query set sets `question_title`, so extraction prompts them with a **question** to continue. That shape never appeared in training, so behavioral distances are measured out-of-distribution. Same signature as the nomic prefix bug: no error, fluent output, effect invisible until measured against a correctly-formed input. **Not a one-line fix** — `ClassDatasetEntry.text_field` is a single string selecting one column at read time, so question+answer is not expressible in YAML and needs a *composition* mechanism in the recipe layer. **Blast radius:** it changes `recipe_hash`, so every affected draw re-materialises and all 25 adapters need retraining. **It splits the levels:** dataset embeddings also use `best_answer`, so if training moves to combined text and the dataset level does not follow, "what a 75/25 mixture means" stops being the same thing at each level — decide both together. **It revives item 2's parked question:** if `text_field` becomes a composition rather than a column selector, the "it is only a read-time projection" argument for keeping it out of the recipe hash weakens considerably. |
| 10 | Generalize CKA to multi-layer / multi-projection sweeps | [cka_notes.md](cka_notes.md) | Parked on a genuine modeling choice, not on effort. Needs a decision in conversation before any code. The other three distance builders were already generalized — see [frobenius_bw_generalization.md](frobenius_bw_generalization.md) for those proofs. |
| 12 | **Behavioral representations have no `representation:` knob, unlike the dataset level** | this row; see also 8 | **What is on disk today** (verified against `05_generated/5191ad734b81daff/`): `embeddings/{slug}.safetensors` holds `matrix` as **`(n_queries, 768)` float32** — one row per query, each row L2-normalized, no pooling across queries — plus a `_meta_json` blob carrying `model_id`, `taxonomy` and `metadata`. The generated **text is not in the tensor file**: it lives beside it in `generations/{slug}.json`, and `GeneratedTextCache.load()` folds it back into `metadata["generated_texts"]` so callers see the original object. At `n_queries=64` that is ~192 KB of tensor + ~33 KB of text per model, growing linearly in `n_queries`. **The asymmetry:** `DatasetEmbeddingTaxonomy` takes `representation: mean\|matrix` and the populated `02_dataset_embeddings` all use `mean` (a `(1, d)` centroid), but `BehavioralTaxonomy` has no such parameter — it always stores the full matrix. **Why that matters less than it looks:** `CosineDistanceMetric` flattens both matrices before comparing, and because row *i* is query *i* for every model and every row is unit-norm, the flattened cosine reduces exactly to the **mean per-query cosine similarity** — `cos(A_flat, B_flat) = (1/n)·Σᵢ⟨aᵢ,bᵢ⟩`. So the metric already averages over queries; keeping the full matrix costs disk but is what makes per-query analysis (centering, subsetting, per-query variance) possible at all — and those are the levers if the signal stays weak. **Decide:** whether behavioral gains a `representation:` option for symmetry, and whether a pooled `(1, 768)` variant is stored alongside so behavioral and dataset centroids are directly comparable. Note a pooled behavioral vector is **not** the same as the flattened cosine — averaging vectors then comparing differs from comparing then averaging. |

| 13 | **The two inference caches diverge on three axes** | this row; see also 9 | `GeneratedTextCache` (`05_generated`) and `ActivationCache` (`04_activations`) store the same *kind* of thing — a per-model representation over a shared query draw — with three different conventions. Recorded now, while the reason for each is still known. **(a) Top-level key:** behavioral is run-wise (`{config_hash}/`), functional is model-wise (`{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/`). Functional needs model-wise because one forward pass yields every layer, making layer choice a read-time decision; behavioral has no such axis, so run-wise costs it nothing. Not obviously worth unifying. **(b) Model key — this is the one with a real failure mode.** `generated_text_cache.model_slug` hashes the **full absolute path** of the adapter directory, so a behavioral entry is keyed to *where the adapter happened to live on disk*: **moving the cache root silently orphans every behavioral representation** — every write still succeeds, and the cache reads as empty. `ActivationCache` follows the `LoRACache` convention (`_slug(base_model_id) / adapter dir name`) and does not have this property. Fixing behavioral means a migration, since the slug is a directory name. **(c) Query record:** behavioral duplicates the full query **text** into `queries.json`; functional stores the `query_key` plus source row **indices**, since `01_datasets` is canonical and `(recipe_hash, n_samples, seed)` determines the text completely. Functional's is the better convention and behavioral's is redundant, but the redundancy is harmless. **Decide:** whether (b) is worth a migration on its own — it is the only one of the three that can lose data. |

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
only test harness — there is no pytest. It registers 53 checks (43 synthetic, 10 that
read the real cache); the data-backed ones report `SKIP` when a path is missing
rather than passing vacuously, so they fail loudly on a missed migration. Run it
before and after any of the items above and compare the pass/skip profile.

`python scripts/verify_sampled_cache.py --full` is the second instrument, and the one
that matters for the dataset cache specifically: draws are stored as source indices, so
it rehydrates each one and checks it against a recorded `rows_sha256`. Nothing else
would notice if the upstream HuggingFace dataset changed underneath. `--fast` checks
revisions and row counts only and takes seconds.

The full run is ~7 s and ~220 MB, so it is cheap enough to run freely. Four flags
narrow it further when you want a fast loop:

    --list              print the check names, run nothing
    -k PATTERN          only checks whose description contains PATTERN
    --data-only         only the 10 that read the real cache — what catches a
                        broken path after a migration
    --synthetic-only    skip the cache entirely
    --include-gpu       additionally run the [gpu] tier, which loads a real model
                        onto a CUDA device. Off by default; the SLURM job passes it
                        after extraction, while the GPU is still allocated.

Run it in the project conda environment (`conda activate taxonomy-env`, the same one
the SLURM scripts activate) — `numpy` is not installed in the base env, so the
checks fail at import there. Baseline as of 2026-08-03, after the functional work:
**53 passed, 0 failed, 0 skipped** across 53 registered checks (43 synthetic,
10 `[data]`), plus 2 `[gpu]` checks behind `--include-gpu`. Both inference levels are
now populated, so nothing skips; the `[data]` checks for each must **skip**, never
pass, whenever their stage directory is absent. Earlier baselines: 45/0/1 after the
behavioral work (the one skip being functional's absent `04_activations/`), and
40/0/0 before either. `t_scan_cache` reports 25 adapters,
**25 with recipes, 25 usable**, and `verify_sampled_cache --fast` reports 521 draws ok.

The jump from 20 usable to 25 is task 2's doing, not a change in the data: the five
oldest adapters were trained under the pre-rename naming (`yahoo_25t0_75t1`) and used to
dangle, because their recipe hash matched nothing in the shared cache. Content-addressing
merged them with their zero-padded twins, so they now resolve.
