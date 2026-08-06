# `02_dataset_embeddings` hid the draw inside a hash — item 15

**Status: done.** Implemented and migrated on 2026-08-06. This note is now the
record of what changed and why, and of what the original plan got wrong. The
reasoning is kept because it still explains the shape of the result.

Companion notes: [`sampled_dataset_storage.md`](sampled_dataset_storage.md) for
the content-addressing change that forced this, [`gram_and_cka.md`](gram_and_cka.md)
§3 for the `gram` redefinition named in §6, and
[`functional_behavioral.md`](functional_behavioral.md) for the inference stages
this aligned to.

---

## 1. How each stage addressed a draw, before

Verified on disk, not inferred:

| stage | how a draw was addressed |
|---|---|
| `01_datasets` | a **file** — `{recipe_hash}/n{n}_s{seed}.json`, **unpadded seed** |
| `02_dataset_embeddings` | **neither** — folded into `{embedder_hash}/` |
| `04_activations` | a **directory** — `.../{recipe_hash}/n{n}_s{seed:02d}/` |
| `05_generated` | a **directory** — same shape as `04`, since PR #7 |

> ⚠️ **The original version of this note claimed 01, 04 and 05 agreed.** They did
> not. `01` wrote `n100000_s0.json` and `04`/`05` wrote `n64_s00`: one coordinate,
> two spellings, and nothing ever compared them — `DrawKeyedCache.draw_name` even
> documented its token as "matching the draw filenames in `01_datasets`" while
> `01` wrote something else. There was no convention to conform to. That is why
> the fix below unifies three stages rather than moving `02` onto an existing one.

## 2. Why `02` was that way, and it was a real fix

`DatasetEmbeddingCache.embedder_hash(embedder_config, representation, n_samples, seed)`
folded all four axes into one opaque 16-hex name.

The reason was sound. Before item 2, a recipe directory was named with the draw
baked in, so the draw was distinguished one level up. Once `recipe_hash` became
content-addressed over `{recipe_type, datasets}`, `_s{seed}` left the recipe
*name* — and seed had to go somewhere. Without it, every seed of a mixture
collapses onto one entry and a seed sweep silently reads a single draw for all
seeds, reporting a variance of zero that is an artifact of the cache rather than
a property of the data.

It went into the hash rather than the path. That fixed the collapse. It was the
placement, not the fix, that this item changed — **the guarantee moved, it did
not go away.**

## 3. What the cache actually contained

All 520 entries enumerated:

| axis | distinct values |
|---|---|
| `representation` | 1 — `mean` (520/520) |
| `embedder_config.model_name` | 1 — `nomic-ai/nomic-embed-text-v1.5` (520/520) |
| `embedder_config.prompt_prefix` | 1 — absent (520/520) |
| `matrix` shape / dtype | 1 — `(1, 768)` float32 (520/520) |
| `n_samples` | **17** |
| `seed` | **10** |

Per-recipe counts: `932b0a7b3595ac99` 135, `2a28fc6d2884d74a` 103, and 94 each
for `6709c078a6cd9b35`, `89bbaff3b7a4e6cb`, `f48e8bb62aceef19`.

**So a directory named `embedder_hash` was in practice a draw hash.** The
embedder axis had never varied. 520 opaque names encoded nothing but
`(n_samples, seed)` — exactly what every other stage wrote in plain text.

The uniform absent `prompt_prefix` is the expected footprint of the nomic-prefix
fix: these are all pre-fix entries, geometrically equivalent but unreachable
under the new `embedder_hash` by design. See
[`embedder_task_prefixes.md`](embedder_task_prefixes.md). It is not a second bug.

## 4. What was built

```
01_datasets/{recipe_hash}/n{n}_s{seed:02d}.json          ← seed now padded

02_dataset_embeddings/{recipe_hash}/
    recipe.json                                          ← stays at recipe level
    n{n}_s{seed:02d}/
        {embedder_hash}/                                 ← embedder_config ALONE
            config.json
            surrogates/{surrogate_hash}/
                config.json                              ← {"representation": ...}
                surrogate.safetensors
```

`src/cache/_draw.py` now owns the draw token for every stage. Writing is narrow
(`draw_name` always pads); reading is wide (`parse_draw_name` accepts both), so
pre-migration names still read and a rename never becomes data loss.

`embedder_hash` dropped to `{embedder_config}` and `representation` moved into
the surrogate spec, so `02` has the same shape as `04`/`05`. `surrogate_hash` is
`DrawKeyedCache.config_hash` rather than a private peer — two hashing schemes for
one concept is exactly how the draw token drifted.

**The code payoff the original note missed:** `_embedder_choice` collapsed from
~50 lines to a set intersection. It only ever did signature bookkeeping because
the hash bundled `n_samples`, which meant demanding one shared hash also demanded
one shared sample count — ruling out the pooled comparison across the size sweep.
Removing `n_samples` from the key removed the reason for the workaround.

`dataset_selector` was added alongside `behavioral_selector` and
`functional_selector`, because `representation` left the hash and `embedder_hash=`
alone can no longer express which representation to read. It is one more axis
`CollectionCache.collection_hash()` cannot see — that is item 14.

### The one semantic difference from `04`/`05`, stated rather than hidden

At `04`/`05` a surrogate is a **read-time view derived from a stored base
artifact**: the raw activations are on disk, so a new view costs only CPU. `02`
has no base artifact. `representation` is chosen *before* embedding and only the
result was ever stored, because the true base — the full `(N, 768)` per-element
embeddings — would cost **6.1 GB** (against 2.1 MB) and a GPU re-embed of
**1,984,744 texts**, and `mean` is not invertible.

So **a `02` surrogate is authored, not derived.** The directory shape matches
exactly; the guarantee behind it is weaker. Adding a representation here means
re-embedding, not a read-time rebuild. Recorded in the class docstring, because a
reader who assumes otherwise will assume wrong.

## 5. Migration, and what it cost

`scripts/migrate_dataset_embedding_layout.py`, additive with a separate
`--prune`, following `migrate_recipe_identity.py` and
`migrate_behavioral_layout.py`. No GPU, no recomputation, no distance changed;
every field the new paths needed was already in each `config.json`, so
destinations were **read off what is stored, never recomputed from the
sampler** — the rule `TODO.md` records under "do not simplify these".

Applied to the shared cache:

| stage | planned | verified byte-identical | pruned |
|---|---|---|---|
| `01_datasets` | 523 draws | 523 | 523 |
| `02_dataset_embeddings` | 520 entries | 520 | 520 |

All 523 seeds were single-digit, so padding produced **zero collisions**; no two
`02` entries shared a `(recipe_hash, n_samples, seed)` key, so the relayout was
1:1. Between apply and prune, `verify_sampled_cache.py --full` rehydrated all
**1046** draws — both spellings — against their recorded `rows_sha256`: 0 failed,
364 s. After pruning: 523 ok.

`check_analysis.py` went **57 → 60 passed**, with the same 2 pre-existing
failures throughout (PR #7's behavioral migration has not been applied to the
shared cache, so the old run-wise `05_generated/{config_hash}/` directories
survive — unrelated to this item).

Three checks added and one inverted:

- `t_one_draw_name` — `01`, `04` and `05` must produce the same token. It
  **failed** against the unpadded layout, which is how the drift was confirmed
  rather than assumed.
- `t_surrogate_hash_shared` — `02` and the inference caches must agree on one
  spec digest.
- `[data] t_dataset_embedding_layout` — nothing of the old shape may survive.
- `t_embedder_hash_seed` was **inverted**: it asserted seed was *in* the hash,
  which is now false by design, so it asserts the same guarantee in its new home.

## 6. Two corrections to TODO item 12, found while verifying — both applied

Both were in `src/taxonomy/dataset_embedding.py`, documentation drift rather than
behavioural bugs:

1. **`representation` has three modes, not two.** Item 12 said
   `representation: mean|matrix`; the signature is
   `Literal["matrix", "gram", "mean"]`. The class docstring opened "Two
   representation modes are available" and documented only `matrix` and `gram`,
   leaving `mean` — the mode every single stored entry uses — undocumented.
2. **The `gram` docstring's cross-reference was stale.** It said the mode
   "Mirrors what `FunctionalTaxonomy` does per layer". That per-layer form was
   **removed**; [`gram_and_cka.md`](gram_and_cka.md) §3 records functional's
   `gram` as redefined so rows are *queries*. The dataset level's
   `(1, N(N+1)/2)` flattened upper triangle no longer mirrors anything.

## 7. A third bug, found while verifying and fixed here

`scan_yahoo_cache` / `scan_yahoo_cache_detailed` (`src/notebook/yahoo_utils.py`)
globbed `*/recipe.json` and regexed the recipe **name** for `_n{n}_s{seed}`.
Content-addressing broke that: each `recipe.json` now holds one arbitrary stale
name — whichever draw wrote it first — so they reported **1 draw per proportion
instead of 94–135**. One stored name, `yahoo_025t0_075t1_s03`, did not match
`YAHOO_RECIPE_RE` at all, so **that proportion vanished entirely**.

Both now list the `n{n}_s{seed}/` directories and derive the proportion from the
recipe's `normalized_class_weights` rather than its name. The class universe is
taken across all recipes, because a *pure* recipe filters to one class and
records only that one — labelling from its own keys alone yields `100t0` where
every other recipe yields `100t0_000t1`, and the two would not group together.

Verified against the cache: 5 proportions, 520 draws, per-proportion counts
135 / 103 / 94 / 94 / 94 — matching §3 exactly.

## 8. What this item did *not* do

The dataset level has no model axis: `taxonomy.recipe_ids()` is passed where the
other levels pass `model_ids`, and `load()` sets `cache_key=""` with the
`_meta_json` `model_id` holding a recipe id. So `02` can never share the
`{base}/{adapter}/` prefix that `03`, `04` and `05` have in common. This aligned
the **draw and surrogate** coordinates only, and that is the whole of what is
alignable.
