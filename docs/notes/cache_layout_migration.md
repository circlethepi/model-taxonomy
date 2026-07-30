# Numbering the shared cache directories

**Status: DONE** (task 1). Applied by `scripts/migrate_cache_layout.py --apply`;
that script remains the record of what moved and carries a `--revert`. This note is
kept for the reasoning, not as a plan.

Two things changed relative to the sketch below, both decided in conversation:

- `adapter_alignments` became **`03A_adapter_alignments`**, not `03_...` — a letter
  suffix marks analysis *of* the objects at that stage rather than a stage of its own.
- The `DiskCache` fallback was **removed from `StructuralTaxonomy`** rather than
  merely left unused. Its only live construction site always supplies a `LoRACache`,
  so the branches were unreachable; structural now has exactly one caching protocol.

Recipe backfill result: 564 of 564 hashes resolved — 562 copied from the
dataset-embedding cache, 2 recovered by re-hashing `results/*/datasets/*.recipe.json`.

Originally deferred so it would not land in the same pass as the comparison
machinery — it touches five cache classes, ~1,700 on-disk directories and the
notebooks, and mixing it with new code is where mistakes come from.

**Strategy chosen:** move the directories and update every path. **No
compatibility shim** — a fallback layer would leave two live layouts and quietly
hide a missed call site.

## Why

The shared cache sits flat, so nothing in the directory listing says which
directories come early in the pipeline and which are derived from them:

```
results/shared_cache/
    adapters/  adapter_alignments/  dataset_embeddings/  representations/  sampled_datasets/
```

A numeric prefix makes the pipeline order readable at a glance. Several
directories may share a number when they sit at the same stage.

## Target layout

| new name | old name | holds | stage |
|---|---|---|---|
| `01_datasets/` | `sampled_datasets/` | sampled rows, `{recipe_hash}/{n}_{seed:010d}.json`; **plus** a mirrored `recipe.json` per hash (see below) | create datasets, and the common test set |
| `02_dataset_embeddings/` | `dataset_embeddings/` | `{recipe_hash}/recipe.json` + `{embedder_hash}/{config.json, embeddings.safetensors}` | dataset taxonomy representations |
| `03_adapters/` | `adapters/` | raw PEFT weights **and** extracted structural reps under `{adapter}/{config_hash}/` | fine-tuning output |
| `03_adapter_alignments/` | `adapter_alignments/` | pairwise Procrustes alignment matrices, `{layer}/{proj}/` | structural taxonomy artifacts (was unnumbered in the original sketch) |
| `04_activations/` | part of `representations/` | functional taxonomy reps + `queries.json` | functional extraction |
| `05_generated/` | part of `representations/` | behavioral taxonomy reps (generated text, embedded) + `queries.json` | behavioral extraction |
| `06_collections/` | `collections/` | per-taxonomy distance matrices, geometries, `index.json` | taxonomy results |

### Three things to know before starting

1. **`representations/` is empty.** It is a `DiskCache` used by the functional and
   behavioral taxonomies. Structural also receives it in `run_taxonomy.py`, but
   `StructuralTaxonomy.extract` checks `LoRACache` first and only falls through to
   `DiskCache` when `lora_cache is None`, which `run_taxonomy.py` never does — so
   structural reps live in `03_adapters/{adapter}/{config_hash}/`. Splitting the
   one `DiskCache` into two instances (`04_activations`, `05_generated`) therefore
   costs **zero migration**: there is nothing in it to move.

2. **Recipes have no shared, hash-indexed home.** They are written to
   `results/<experiment>/datasets/{name}.recipe.json` by `build_datasets.py`, and
   mirrored into `dataset_embeddings/{recipe_hash}/recipe.json` by
   `DatasetEmbeddingCache.save`. So today the *only* way to resolve a
   `recipe_hash` to a recipe is to reach into the dataset-embedding cache — which
   `src/analysis/discovery.py` currently does, and which fails for any recipe that
   was never embedded. Mirroring recipes into `01_datasets/{recipe_hash}/recipe.json`
   fixes that and is why `01` is the right home.

3. **The query/test set:** canonical copy in `01_datasets` (it is an ordinary
   sampled dataset, already keyed by `(recipe_hash, n_samples, seed)`), plus a
   convenience `queries.json` in `04`/`05` with the key recorded in `config.json`
   so the canonical copy stays authoritative.

### Asymmetry worth documenting rather than discovering

Structural representations live *inside* `03_adapters/{adapter}/{config_hash}/`,
beside the weights they were derived from, while functional and behavioral reps
live in their own stage directories. That is defensible — a LoRA rep is a
transformation of that adapter's own weights and nothing else — but it is not
obvious, so `docs/concepts.md` should say it.

## Call sites to change

Verified by grep, current as of this note:

```
src/cache/lora_cache.py:43              self._loras_dir = self.root / "adapters"
src/cache/dataset_embedding_cache.py:36 self._base = self.root / "dataset_embeddings"
src/cache/collection_cache.py:50        self._collections_dir = self.root / "collections"
scripts/_utils.py:397                   DiskCache(cache_dir / "representations")     -> split 04/05
scripts/_utils.py:407                   SampledDatasetCache(cache_dir / "sampled_datasets")
src/analysis/discovery.py:293-295       adapters / dataset_embeddings / sampled_datasets
src/notebook/yahoo_utils.py:97,136      Path(cache_root) / "dataset_embeddings"
scripts/check_analysis.py:43            ADAPTER_ROOT = REPO / "results/shared_cache/adapters"
scripts/check_analysis.py:904,950       (root / "adapters").exists()
src/analysis/comparison.py              Path(root) / "adapters"  (in _structural_matrix)
```

Plus the notebooks, which pass `results/shared_cache/adapter_alignments` into
`src/notebook/structure.py`'s `cache_dir` parameter — the alignment path is built
by the caller, not by a cache class.

### Two traps

- **`src/core/analysis.py:62` and `:102` use `path / "representations"` — do NOT
  rename these.** That is the layout *inside* a saved `TaxonomyAnalysis`
  directory (`results/<exp>/taxonomy/{level}/representations/`), which has nothing
  to do with the shared cache. Renaming it would silently break
  `ModelTaxonomyProfile.load` on every existing saved profile.
- **`src/notebook/lora_weights.py:185`** defaults `adapter_root` to
  `results/shared_cache/peft_adapters`, a directory that does not exist. Every
  caller passes the real path, so the default has never been exercised. Fix it to
  `03_adapters` during the migration rather than leaving a second wrong default.

## Migration sketch

```python
# scripts/migrate_cache_layout.py --dry-run | --apply
RENAMES = {
    "sampled_datasets":   "01_datasets",
    "dataset_embeddings": "02_dataset_embeddings",
    "adapters":           "03_adapters",
    "adapter_alignments": "03_adapter_alignments",
    "collections":        "06_collections",
}
```

`os.rename` within one filesystem is a metadata operation, so this is effectively
instant regardless of the ~1,700 directories. `representations/` is empty, so
delete it rather than moving it, and create `04_activations/` and `05_generated/`
on first write.

Order of work: migrate → update the call sites above → `python
scripts/check_analysis.py` (its data-backed checks read the cache directly and
will fail loudly on a missed path) → run the notebooks' first cells.

## Verification

`scripts/check_analysis.py` is the sharpest instrument here: `t_scan_cache` and
`t_comparison_end_to_end` walk the real cache and will report `SKIP` (path gone)
or `FAIL` rather than passing vacuously. Confirm they still `PASS` with the same
model counts as before the move — 25 adapters, 20 with recipes, 20 usable.
