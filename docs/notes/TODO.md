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

The lettering is for reference in conversation, not a strict execution order.
Dependencies are called out in the State column.

## Items

| # | Item | Note | State |
|---|---|---|---|
| A | Renumber the shared cache to `01_datasets` … `06_collections` | [cache_layout_migration.md](cache_layout_migration.md) | Agreed, fully specced, move-and-update with no compatibility shim. Blocks D; also removes the unpinned-query-set problem in H. |
| B | `scripts/run_comparison.py` + a YAML `comparison:` block | [comparison_followups.md](comparison_followups.md) §1 | One decision open: build a `CacheIndex` from resolved model IDs, or let `comparison.models` accept the same `base_models` / `fine_tuned` tokens the other config sections use. The second is more consistent. |
| C | `src/plots/simplex.py` + `notebooks/4_taxonomy_comparison.ipynb` | [comparison_followups.md](comparison_followups.md) §2 | Five figures specced: ternary/simplex, recovered-vs-true, cross-taxonomy disparity heatmap, Shepard diagram, convergence-in-`n`. Load the `dataviz` skill first, and match `src/plots/config.py`. |
| D | `docs/guides/pipeline_stages.md` — a runnable template YAML per stage | [comparison_followups.md](comparison_followups.md) §3 | Do after A, so the "what it writes to the cache" column can quote the final directory names. |
| E | Add distance correlation (dCor); demote Mantel's p-value to descriptive | [distance_matrix_comparison.md](distance_matrix_comparison.md) §1 | Mantel has inflated type-I error on dependent off-diagonal entries; PROTEST is already implemented and better calibrated. Do **not** drop Mantel — existing figures reference it. |
| F | Fit `T: D_taxonomy → D_simplex` and test whether it converges in `n` | [distance_matrix_comparison.md](distance_matrix_comparison.md) §2 | Research, not plumbing. The data already exists: the `by_seed/` grouping varies `n_samples` at fixed seed, and `compare_taxonomies` already persists a restricted `T_n`. Fit on distance matrices, not raw MDS coordinates. |
| G | Make `src/metrics/` handle variable-length representation rows | in-code TODOs at `src/metrics/cka.py:1`, `src/metrics/frobenius.py:1`, `src/taxonomy/structural.py:135` | Blocks the full structural pipeline through `ModelRepresentation`. Storage format still undecided (zero-pad / per-module-type / a ragged type). Also relevant to H — see its open question 4. |
| H | Functional/behavioral: run them, then wire them into the comparison layer | [functional_behavioral.md](functional_behavioral.md) | Both levels are fully implemented and have never been run; `results/shared_cache/representations/` is empty. Four open design questions and three things to check on first run. |
| I | Generalize CKA to multi-layer / multi-projection sweeps | [cka_notes.md](cka_notes.md) | Parked on a genuine modeling choice, not on effort. Needs a decision in conversation before any code. The other three distance builders were already generalized — see [frobenius_bw_generalization.md](frobenius_bw_generalization.md) for those proofs. |

## Small wins

Recorded so they survive; each is cheap and independent.

- **Drop torch from the distance-computation import path.**
  `src/notebook/lora_weights.py` opens adapters with `safe_open(framework="pt")` and
  immediately calls `.numpy()`. The adapter weights are `float32`, so
  `framework="numpy"` returns the same arrays and removes torch from the dependency
  path — measured 587 MB → ~190 MB of import overhead. Check for `bfloat16` adapters
  first; numpy has no such dtype.
- **Fix the nonexistent default in `load_lora_weights`.**
  `src/notebook/lora_weights.py:185` defaults `adapter_root` to
  `results/shared_cache/peft_adapters`, which does not exist. Every caller passes a
  real path, so it has never been exercised. Fix it during A rather than leaving a
  second wrong default behind.
- **Read the configured device for the sentence embedders.** Both
  `SentenceTransformerEmbedder` construction sites hardcode `device="cpu"` —
  `scripts/_utils.py:469` (behavioral) and `scripts/_utils.py:574`
  (dataset_embedding) — so every sentence embedding the pipeline has produced so
  far, including the populated `dataset_embeddings` cache, was computed on CPU. Fine
  for MiniLM-L6, a bottleneck for anything larger. Fix both together. See H, "Things
  to check", item 1.

## Do not "simplify" these

- **`compare_taxonomies` projects from the largest available dimension, not `k-1`.**
  Projecting from exactly `k-1` makes the anchors' affine hull the whole space, every
  residual becomes identically zero, and the off-simplex diagnostic silently
  disappears. Pinned by `t_projection_dimension_matters` in
  `scripts/check_analysis.py`.
- **`src/core/analysis.py` uses `path / "representations"`, and that is not the
  shared cache.** It is the layout *inside* a saved `TaxonomyAnalysis` directory.
  Renaming it during A would break `ModelTaxonomyProfile.load` on every existing
  saved profile. See the "Two traps" section of
  [cache_layout_migration.md](cache_layout_migration.md).

## Verification

`python scripts/check_analysis.py` is the sharpest instrument in the repo and the
only test harness — there is no pytest. It registers 35 checks (28 synthetic, 7 that
read the real cache); the data-backed ones report `SKIP` when a path is missing
rather than passing vacuously, so they fail loudly on a missed migration. Run it
before and after any of the items above and compare the pass/skip profile.

Run it in the project conda environment (`conda activate taxonomy-env`, the same one
the SLURM scripts activate) — `numpy` is not installed in the base env, so the
checks fail at import there. Baseline as of 2026-07-29: **35 passed, 0 failed, 0
skipped**, with `t_scan_cache` reporting 25 adapters, 20 with recipes, 20 usable.
