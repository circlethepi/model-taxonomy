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

The numbering is for reference in conversation. It is not a strict execution order,
with one exception: **2 is meant to follow 1**. Other dependencies are called out in
the State column.

## Items

| # | Item | Note | State |
|---|---|---|---|
| ~~1~~ | ~~Number the shared cache to `01_datasets` … `06_collections`~~ | [cache_layout_migration.md](cache_layout_migration.md) | **Done.** Migrated by `scripts/migrate_cache_layout.py`; recipes now have a hash-indexed home at `01_datasets/{recipe_hash}/recipe.json`, and the structural `DiskCache` fallback was removed. |
| 2 | Sampled-dataset storage, and whether n/seed belong in recipe identity | [sampled_dataset_storage.md](sampled_dataset_storage.md) | Take this next. The two questions are one question: the row-ordering scheme is unreachable while the recipe name carries `_n{n}_s{seed}`, since that makes every n a different hash. Needs a decision in conversation before code. |
| 3 | `scripts/run_comparison.py` + a YAML `comparison:` block | [comparison_followups.md](comparison_followups.md) §1 | One decision open: build a `CacheIndex` from resolved model IDs, or let `comparison.models` accept the same `base_models` / `fine_tuned` tokens the other config sections use. The second is more consistent. |
| 4 | `src/plots/simplex.py` + `notebooks/4_taxonomy_comparison.ipynb` | [comparison_followups.md](comparison_followups.md) §2 | Five figures specced: ternary/simplex, recovered-vs-true, cross-taxonomy disparity heatmap, Shepard diagram, convergence-in-`n`. Load the `dataviz` skill first, and match `src/plots/config.py`. |
| 5 | `docs/guides/pipeline_stages.md` — a runnable template YAML per stage | [comparison_followups.md](comparison_followups.md) §3 | Unblocked by 1 — the "what it writes to the cache" column can now quote the final directory names. |
| 6 | Add distance correlation (dCor); demote Mantel's p-value to descriptive | [distance_matrix_comparison.md](distance_matrix_comparison.md) §1 | Mantel has inflated type-I error on dependent off-diagonal entries; PROTEST is already implemented and better calibrated. Do **not** drop Mantel — existing figures reference it. |
| 7 | Fit `T: D_taxonomy → D_simplex` and test whether it converges in `n` | [distance_matrix_comparison.md](distance_matrix_comparison.md) §2 | Research, not plumbing. The data already exists: the `by_seed/` grouping varies `n_samples` at fixed seed, and `compare_taxonomies` already persists a restricted `T_n`. Fit on distance matrices, not raw MDS coordinates. |
| 8 | Make `src/metrics/` handle variable-length representation rows | in-code TODOs at `src/metrics/cka.py:1`, `src/metrics/frobenius.py:1`, `src/taxonomy/structural.py` | Blocks the full structural pipeline through `ModelRepresentation`. Storage format still undecided (zero-pad / per-module-type / a ragged type). Also relevant to 9 — see its open question 4. |
| 9 | Functional/behavioral: run them, then wire them into the comparison layer | [functional_behavioral.md](functional_behavioral.md) | Both levels are fully implemented and have never been run in the shared cache; `04_activations/` and `05_generated/` do not exist yet. Four open design questions and three things to check on first run. |
| 10 | Generalize CKA to multi-layer / multi-projection sweeps | [cka_notes.md](cka_notes.md) | Parked on a genuine modeling choice, not on effort. Needs a decision in conversation before any code. The other three distance builders were already generalized — see [frobenius_bw_generalization.md](frobenius_bw_generalization.md) for those proofs. |

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
- **Read the configured device for the sentence embedders.** Both
  `SentenceTransformerEmbedder` construction sites hardcode `device="cpu"` —
  `scripts/_utils.py:469` (behavioral) and `scripts/_utils.py:574`
  (dataset_embedding) — so every sentence embedding the pipeline has produced so
  far, including the populated `02_dataset_embeddings` cache, was computed on CPU.
  Fine for MiniLM-L6, a bottleneck for anything larger. Fix both together. See 9,
  "Things to check", item 1.

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
- **`{output_dir}/adapters/` is deliberately unnumbered.** The numeric prefixes
  describe shared-cache pipeline stages; an experiment output directory has no such
  ordering, and renaming it would disturb the `LoRACache` slugs in the legacy
  per-experiment caches, which encode the path literally.

## Verification

`python scripts/check_analysis.py` is the sharpest instrument in the repo and the
only test harness — there is no pytest. It registers 35 checks (28 synthetic, 7 that
read the real cache); the data-backed ones report `SKIP` when a path is missing
rather than passing vacuously, so they fail loudly on a missed migration. Run it
before and after any of the items above and compare the pass/skip profile.

The full run is ~7 s and ~220 MB, so it is cheap enough to run freely. Four flags
narrow it further when you want a fast loop:

    --list              print the check names, run nothing
    -k PATTERN          only checks whose description contains PATTERN
    --data-only         only the 7 that read the real cache — what catches a
                        broken path after a migration
    --synthetic-only    skip the cache entirely

Run it in the project conda environment (`conda activate taxonomy-env`, the same one
the SLURM scripts activate) — `numpy` is not installed in the base env, so the
checks fail at import there. Baseline as of 2026-07-30, after task 1: **35 passed,
0 failed, 0 skipped**, with `t_scan_cache` reporting 25 adapters, 20 with recipes,
20 usable — unchanged from before the cache migration.
