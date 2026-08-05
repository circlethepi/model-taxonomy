# `02_dataset_embeddings` hides the draw inside a hash

Written because every other cache stage spells the draw out in the path and this
one does not, and because the directory named `embedder_hash` turns out — on the
data actually stored — to be keyed by almost nothing but the draw. The current
layout is not a mistake; it was a deliberate fix for a real bug. But the reason
it was needed has a better answer available now, and the cost is paid every time
someone tries to read the cache by eye.

Companion notes: [`sampled_dataset_storage.md`](sampled_dataset_storage.md) for
the content-addressing change that forced this, and
[`gram_and_cka.md`](gram_and_cka.md) §3 for the `gram` representation named in §5.

---

## 1. How each stage addresses a draw

Verified on disk, not inferred:

| stage | how a draw is addressed |
|---|---|
| `01_datasets` | a **file** — `{recipe_hash}/n{n}_s{seed}.json` |
| `02_dataset_embeddings` | **neither** — folded into `{embedder_hash}/` |
| `04_activations` | a **directory** — `{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/` |
| `05_generated` | a **directory** — same shape as `04` |

There are three schemes here, not two. 01 uses a filename where 04/05 use a
directory of the same name, and that difference is forced rather than stylistic:
a draw at 04/05 has to hold several artifacts (`activations`/`embeddings`,
`runs`, `surrogates`), so it must be a directory, whereas 01 stores one JSON per
draw and a filename suffices. The rhyme is deliberate — `ActivationCache.draw_name`
carries a comment saying `n{n}_s{seed}` matches the draw filenames in `01_datasets`.

`02` is the one that does neither. Its tree is exactly two levels deep:

```
02_dataset_embeddings/{recipe_hash}/
    recipe.json                 ← human-readable, shared across embedder configs
    {embedder_hash}/
        config.json             ← embedder_config + representation + n_samples + seed
        embeddings.safetensors  ← (1, 768) f32 + _meta_json
```

5 recipe directories, 520 embedder directories, and **zero paths matching
`n*_s*` anywhere**.

## 2. Why it is that way, and it was a real fix

`DatasetEmbeddingCache.embedder_hash(embedder_config, representation, n_samples, seed)`
folds all four axes into one opaque 16-hex name.

The reason is recorded in its docstring and it is sound. Before item 2, a recipe
directory was named with the draw baked in, so the draw was already distinguished
one level up. Once `recipe_hash` became content-addressed over
`{recipe_type, datasets}`, `_s{seed}` left the recipe *name* — and seed had to go
somewhere. Without it, every seed of a mixture collapses onto one entry and the
seed sweep silently reads a single draw for all seeds, reporting a variance of
zero that is an artifact of the cache rather than a property of the data.

It went into the hash rather than the path. That fixed the collapse. It is the
placement, not the fix, that this note questions.

## 3. What the cache actually contains

All 520 entries enumerated:

| axis | distinct values |
|---|---|
| `representation` | 1 — `mean` (520/520) |
| `embedder_config.model_name` | 1 — `nomic-ai/nomic-embed-text-v1.5` (520/520) |
| `embedder_config.prompt_prefix` | 1 — absent (520/520) |
| `matrix` shape / dtype | 1 — `(1, 768)` float32 (520/520) |
| `n_samples` | **17** |
| `seed` | **10** |

Per-recipe counts: `2a28fc6d2884d74a` 103, `932b0a7b3595ac99` 135, and 94 each
for `6709c078a6cd9b35`, `89bbaff3b7a4e6cb`, `f48e8bb62aceef19`.

**So a directory named `embedder_hash` is in practice a draw hash.** The embedder
axis has never varied. 520 opaque names encode nothing but `(n_samples, seed)` —
exactly what every other stage writes in plain text in the path.

The uniform absent `prompt_prefix` is the expected footprint of the nomic-prefix
fix: these are all pre-fix entries, geometrically equivalent but unreachable
under the new `embedder_hash` by design. See
[`embedder_task_prefixes.md`](embedder_task_prefixes.md). It is not a second bug.

**The cost is legibility.** `ls` cannot answer "which draws are embedded?".
`list_embedder_configs` needs one JSON parse per entry to find out, and
`list_embedder_hashes` exists purely to dodge that — but returns names that are
still opaque. Contrast `04_activations`, where the same question is a directory
listing.

## 4. Proposed layout

```
02_dataset_embeddings/{recipe_hash}/n{n}_s{seed}/{embedder_hash}/
```

with `n_samples` and `seed` **dropped from the hash signature**, so it keys
`{embedder_config, representation}` only — which is what its name claims.

The prefix then matches `01_datasets` exactly and the draw component matches
`04`/`05`.

**Blast radius is large but cheap.** The signature change re-keys all 520
entries, but the migration is content-preserving and needs **no GPU and no
recomputation**: each `config.json` already records `n_samples`, `seed`,
`representation` and `embedder_config`, so the new path is fully recoverable from
what is on disk. Follow the rule in `TODO.md`'s "do not simplify these" —
recover by content-matching what is stored, never by re-running the sampler.

**What this item does *not* do.** The dataset level has no model axis at all:
`taxonomy.recipe_ids()` is passed where the other levels pass `model_ids`, and
`load()` sets `cache_key=""` with the `_meta_json` `model_id` holding a recipe id.
So `02` can never share the `{base}/{adapter}/` prefix that `03`, `04` and `05`
have in common. This aligns the **draw** coordinate only, and that is the whole
of what is alignable.

## 5. Two corrections to TODO item 12, found while verifying

Both are in `src/taxonomy/dataset_embedding.py`, and both are documentation
drift rather than behavioural bugs:

1. **`representation` has three modes, not two.** Item 12 says
   `representation: mean|matrix`; the signature is
   `Literal["matrix", "gram", "mean"]`. The class docstring is also stale — it
   opens "Two representation modes are available" and then documents only
   `matrix` and `gram`, leaving `mean` undocumented. `mean` is the mode every
   single stored entry uses.

2. **The `gram` docstring's cross-reference is stale.** It says the mode
   "Mirrors what `FunctionalTaxonomy` does per layer". That per-layer form was
   **removed** — [`gram_and_cka.md`](gram_and_cka.md) §3 records functional's
   `gram` as redefined so that rows are *queries*, and lists the old stacked
   per-layer triangles as removed. The dataset level's
   `(1, N(N+1)/2)` flattened upper triangle no longer mirrors anything.

Neither changes a stored number. Fix the docstrings when touching this file.
