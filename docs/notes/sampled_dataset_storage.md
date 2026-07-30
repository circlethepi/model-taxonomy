# Sampled-dataset storage, and whether n/seed belong in recipe identity

**Status:** agreed as the item to take immediately after the cache renumbering
(task 1). Two questions that turned out to be the same question, so they are filed
together — the storage win is unreachable while recipe identity stays as it is.

## The finding that reframes it

`recipe_hash` is a SHA-256 of `{name, datasets}` (`src/datasets/recipe.py`,
`DatasetRecipe.recipe_hash`), and the name already carries `_n{n}_s{seed:02d}`,
appended by `expand_dataset_n_samples` and `expand_dataset_seeds` in
`scripts/_utils.py`. So n=100 and n=1000 of the same mixture are **different recipe
hashes**.

Confirmed on disk before the migration: 564 hashes under `sampled_datasets/`, 564
files, every hash holding exactly one. `SampledDatasetCache`'s `(recipe_hash,
n_samples, seed)` key is effectively single-valued — the n and seed in the filename
are decoration, and there is no cross-n reuse to exploit.

This may well be deliberate: `DatasetEmbeddingTaxonomy` treats "this mixture at
n=100, seed 0" as its own point in the taxonomy, and it needs distinct identities to
do that. The question is whether identity-for-the-taxonomy and
identity-for-the-cache should be the same string.

## Is the cache worth keeping at all?

Yes, on two counts.

- **Reuse across stages.** The rows are read by at least `scripts/finetune_lora.py`
  and `scripts/extract_reprs.py` / `scripts/run_taxonomy.py`, plus every re-run. A
  miss costs `load_dataset` + HF `.shuffle(seed)` + `.select`, then an O(n) Python
  loop materialising `dict(row)` per row (`src/datasets/mixed_dataset.py`,
  `MixedDataset._load`). That last part is the real cost at n=265,000. It is *not*
  avoiding a download — HuggingFace has its own disk cache.
- **Provenance.** It pins the exact rows a LoRA was trained on, independent of the
  upstream HF dataset changing underneath.

**Cost:** 2.1 GB, heavily skewed. The 26 files with n ≥ 20,000 account for 1.6 GB
(74%); everything at n ≤ 2,000 — the entire n-sweep working set — is 133 MB. With
28 TB free this is not yet pressure, but it grows with the large-n sweeps.

## The row-ordering proposal

Store one ordering per (mixture, seed); n takes the first n rows. Viable, and it
composes with the existing proportion semantics — but not against the current
sampler, and one property has to be designed in rather than assumed.

- **The per-entry half is already prefix-nested.** `ds.shuffle(seed=self.seed)` then
  `.select(range(count))` is a prefix, and `count` is monotone in n.
- **The merge is not.** `idx = rng.permutation(len(all_samples))` draws a *different*
  permutation for each n, because the length depends on n. So the first 100 rows of
  the n=1000 draw ≠ the n=100 draw.
- **The constraint that matters.** A plain shuffled prefix preserves mixture weights
  only in expectation. The current code guarantees them *exactly* at every n via
  `_allocate_counts` (largest-remainder). At n=10 with 50/50 weights a plain prefix
  can easily give 7/3 — and n=1,2,5,10,20 are all in the cache, so small n is exactly
  where the sweep lives.
- **The fix.** Build the merge order by incremental largest-deficit interleaving:
  at each step take from the entry whose `target_count(k) - taken` is largest. Every
  prefix then has exactly the counts `_allocate_counts` would return, so "first n
  rows" is exact and the scheme is fully compatible with what the sampler promises
  today.

Two consequences to accept going in:

1. Changing the sampler changes which rows future draws produce, so adapters already
   trained stop being reproducible by the new code. Version the sampler or accept it.
2. The storage win only materialises once n and seed leave recipe identity — today
   there is one file per hash and nothing to dedupe.

## Cheaper options that need no key change

- **gzip the JSON** — 3–5× on this text, `put`/`get` become `gzip.open`.
- **Parquet instead of JSON** — 5–10×, and faster to load than `json.loads`.
- **Store source row indices instead of rows** — ~100×, at the cost of depending on
  the upstream HF dataset again, which is the provenance property the cache exists to
  provide. Would need a dataset fingerprint stored alongside.

## Settle first

Whether n and seed belong in recipe identity. Removing them enables the ordering
scheme and collapses 564 near-duplicate recipes to one per mixture, but rewrites every
recipe hash — orphaning all 562 dataset-embedding entries and the `recipe_hash`
references in 25 adapters' `experiment_meta.json`. That is a migration on the scale of
task 1, and it needs a decision in conversation before any code.
