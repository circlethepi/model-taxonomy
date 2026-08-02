# Functional and behavioral taxonomies: details still to work out

**Status:** both levels are fully implemented and fully wired into the scripts, and
neither has ever been run. Nothing below is a bug report — these are the details to
settle and the things to look at when the first extraction happens.

This note is written to be readable without the code open. Components are named and
described where they first come up.

## Contents

- [What exists, and what has never run](#what-exists-and-what-has-never-run)
- [Open questions to settle](#open-questions-to-settle)
- [Things to check when it first runs](#things-to-check-when-it-first-runs)
- [The query set is not pinned](#the-query-set-is-not-pinned)
- [Smallest run that would settle this](#smallest-run-that-would-settle-this)

---

## What exists, and what has never run

Two taxonomy levels compare models by running inference on a shared set of query
strings:

- **`FunctionalTaxonomy`** (`src/taxonomy/functional.py`) — compares models by their
  *internal activations*. For each query and each selected layer it pools the hidden
  states to one vector.
- **`BehavioralTaxonomy`** (`src/taxonomy/behavioral.py`) — compares models by their
  *generated text*. Each query's continuation is embedded with a sentence embedder,
  giving one vector per query.

Both are complete implementations, constructed by `make_functional_taxonomy` /
`make_behavioral_taxonomy` in `scripts/_utils.py`, driven by
`scripts/extract_reprs.py` (which caches representations) and
`scripts/run_taxonomy.py` (which turns them into distance matrices), and configured
under `extraction.taxonomies` in `experiments/example.yaml:79-102`.

What is missing is that **nothing has ever populated their cache**. They write to
separate `DiskCache` instances rooted at `results/shared_cache/04_activations/` and
`05_generated/` (one flat `representations/` before the cache renumbering), and
neither directory exists yet. Both are provisional homes — when these levels are
built out they are expected to grow their own cache classes the way structural has
`LoRACache`, at which point these are the natural directories for whatever those
classes write. Three consequences follow, and all three are about the absence of data rather
than about the extraction code:

**1. The comparison layer cannot read them.** The *comparison layer* is
`src/analysis/comparison.py`, the module that scores a taxonomy's geometry against
the known data-mixing proportions. Its three entry points are
`build_taxonomy_artifacts` (obtain a distance matrix for one taxonomy level, from
cache or computed), `compare_taxonomies` (score those geometries against the
ground-truth simplex), and `compare_all_slices` (do that across model groupings and
write `report.json` / `report.md`). Internally, `_compute_distance_matrix` has a
branch per level; `structural` and `dataset_embedding` are implemented, and
functional and behavioral raise instead:

```python
# src/analysis/comparison.py:284
if taxonomy in ("functional", "behavioral"):
    raise NotImplementedError(
        f"the {taxonomy!r} taxonomy has no cached representations to read. ..."
    )
```

Adding a `_functional_matrix` / `_behavioral_matrix` pair beside the existing two is
the shape of the fix — but see open questions 1 and 2 first, because the interface
they need is not settled.

**2. Cache discovery cannot select for them.** `scan_cache`
(`src/analysis/discovery.py`) walks the shared cache and returns a `CacheIndex` of
`CacheEntry` objects, one per model. Each entry carries an `available` dict whose
keys are *availability tokens* — fixed strings naming an artifact that model either
has or lacks:

```python
# src/analysis/discovery.py:47
_TAXONOMY_AVAILABILITY = (
    "structural_weights",
    "structural_repr",
    "dataset_embedding",
    "sampled_rows",
)
```

`CacheIndex.with_available(*tokens)` filters to the models that have all the named
tokens, which is how a comparison selects a usable model set. There is no token for
functional or behavioral, so there is no way to ask "which models have functional
representations". Open question 1 is about what such a token would even mean.

**3. Test coverage: behavioral now, functional still not.**
`scripts/check_analysis.py` is the repo's verification script and its only test
harness — there is no pytest. It registers checks in three tiers: **35 synthetic,
9 that read the real cache (`[data]`), and 1 that needs a GPU (`[gpu]`)**. The GPU
tier is off unless `--include-gpu` is passed, because it loads a multi-GB model;
the SLURM job passes it after extraction, while the device is still allocated.

Behavioral is covered: cache round-trip, config-hash stability, the
`padding_side="left"` pin, representation well-formedness, batch invariance, and a
row in `t_comparison_end_to_end`. **Functional still has none**, and the only
mention of it remains a fallback that picks whichever model-level taxonomy happens
to be present.

(The figures above are counted from the lists themselves — re-count
`len(SYNTHETIC)` / `len(DATA_BACKED)` / `len(GPU_BACKED)` rather than trusting this
sentence, which has gone stale once already: it read "35 checks (28 synthetic, 7)"
for some time after the cache migration added five.)

---

## Open questions to settle

### 1. How should cache discovery report availability for a config-keyed cache?

The four existing availability tokens are answerable from the filesystem alone. For
example `structural_weights` is just a file-existence test
(`src/analysis/discovery.py:347`):

```python
"structural_weights": (adapter_dir / "adapter_model.safetensors").exists(),
```

Functional and behavioral representations are not answerable that way, because of
how `DiskCache` names files. `DiskCache.key_for(model_id, config)`
(`src/cache/disk.py:115`) hashes the *entire* config dict, then hashes the model id
against that hash, and the result becomes the filename:

```python
config_hash = hashlib.sha256(repr(sorted(config.items())).encode()).hexdigest()[:16]
return hashlib.sha256(f"{model_id}::{config_hash}".encode()).hexdigest()[:16]
# → {cache_dir}/{key[:2]}/{key}.safetensors
```

The config for these two levels includes the full list of query strings, so the key
depends on every query. Nothing about the model or the taxonomy is recoverable from
the path — those are recorded *inside* the file, in a `_meta_json` entry alongside
the matrix. So you cannot tell whether a representation exists for a model without
already holding the exact config that produced it.

Three ways out:

| option | how | cost |
|---|---|---|
| **(a) walk and read** | `scan_cache` opens each file in the `DiskCache`, reads its `_meta_json`, and builds a model → taxonomy inventory | needs no config, keeps `with_available` uniform; but "available" then means "some rep exists", not "the rep for *this* config exists" |
| **(b) config-aware check** | pass a config in and test `exists(key_for(model_id, config))` | exact; but changes the `with_available` signature, which every current caller relies on |
| **(c) bypass discovery** | the comparison layer takes the taxonomy config directly and never asks the index | no change to discovery; but these two levels then get selected differently from the other two |

**(a) looks best**, because keeping `with_available` uniform matters more than
precision here, and the metadata needed is already in every file. It should be
documented honestly: for these two levels, "available" answers a coarser question
than it does for the other four tokens.

### 2. Which functional representation mode feeds the comparison, and how is it reduced?

`FunctionalTaxonomy` has a `representation` setting with two options that produce
differently-shaped matrices, and — importantly — differently-*meaning* rows
(`src/taxonomy/functional.py:143-161`):

| mode | shape | rows are | intended metric |
|---|---|---|---|
| `gram` (default) | `(n_layer_vecs, N_queries·(N_queries+1)/2)` | **layers** | CKA |
| `matrix` | `(N_queries, n_layer_vecs · hidden_dim)` | **queries** | Frobenius or CKA |

`gram` stacks the upper triangle of each layer's query-by-query Gram matrix, so a
row is one layer's similarity structure over the queries. `matrix` concatenates a
query's per-layer activation vectors, so a row is one query.

The comparison layer needs one point per model. Reducing a matrix whose rows are
layers is a different operation from reducing one whose rows are queries, so the two
modes are **not interchangeable** and a single reduction cannot serve both. This
needs a deliberate decision, not a default.

### 3. How do layer indices line up across taxonomy levels?

Two levels both talk about "layers" and mean different numbering:

- **Functional** takes `layer_indices` as indices into the model's `hidden_states`
  tuple, which has length `n_layers + 1` — index 0 is the embedding output, and the
  config default is negative (`[-1, -4, -8]`, counting from the end).
- **Structural** uses absolute adapter layer numbers; the worked comparison examples
  use `layers=[27]`.

Neither is wrong, but they are different schemes, so any cross-level statement of
the form "the same layer in both taxonomies" requires an explicit mapping that does
not currently exist anywhere.

### 4. What does routing these levels through `src/metrics/` expose?

Today the comparison layer never calls `src/metrics/`. For the structural level it
routes through the low-rank builders in `src/notebook/structure.py`, which compute
distances directly from the LoRA `A`/`B` factors without materialising the weight
delta — see `frobenius_bw_generalization.md` for why that is exact.

Functional and behavioral have no such factorised shortcut: they produce dense
matrices, so distances between them must go through `src/metrics/` (`cka.py`,
`frobenius.py`, `vector.py`). That makes wiring them in the **first** thing to call
`src/metrics/` from the comparison layer, and those modules carry a known
limitation — they assume `ModelRepresentation.matrix` has uniform row lengths
(`src/metrics/cka.py:1`, `src/metrics/frobenius.py:1`; tracked as item 8 in
`TODO.md`).

That limitation is benign *for these two levels*, whose matrices are uniform by
construction. The point is that item 8 stops being purely a structural-pipeline
concern once this path exists, so the two items should be sequenced with that in
mind.

---

## Things to check when it first runs

Ordered by how confident the diagnosis is.

### 1. The sentence embedders are pinned to CPU

`make_behavioral_taxonomy` builds its sentence embedder with `device` as a literal,
ignoring the `extraction.device` setting that every other component reads
(`scripts/_utils.py:469`):

```python
embedder = SentenceTransformerEmbedder(
    model_name=ecfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
    device="cpu",                      # <- not read from extraction.device
    ...
)
```

This is not specific to the behavioral level — `make_dataset_embedding_taxonomy`
does the same thing at `scripts/_utils.py:574`. Both are the only two
`SentenceTransformerEmbedder` construction sites in the repo, so **every** sentence
embedding this pipeline has ever produced was computed on CPU, including the
populated `dataset_embeddings` cache.

Harmless for MiniLM-L6, a hard bottleneck for any larger embedder. This one is
unambiguous and the fix is one line each plus a config default; do both sites
together, since fixing only the behavioral one leaves the asymmetry in place.

### 2. Functional does not pin the tokenizer's padding side; behavioral does

**The mechanism.** `tokenizer(..., padding=True)` pads every sequence in a batch out
to the length of the longest one in that batch. `tokenizer.padding_side` decides
whether those pad tokens are appended *after* the real tokens (`"right"`) or
prepended *before* them (`"left"`).

**The asymmetry.** `BehavioralTaxonomy` pins it
(`src/taxonomy/behavioral.py:105`):

```python
tokenizer.padding_side = "left"
```

`FunctionalTaxonomy` does not (`src/taxonomy/functional.py:111`) — it accepts
whatever the checkpoint's `tokenizer_config.json` specifies, falling back to the
transformers class default of `"right"` when the checkpoint leaves it unset. So the
padding side for functional extraction is **model-dependent and unpinned**, which is
the actual defect: the same code gives different behaviour for different checkpoints.
(Checked on the one Llama `tokenizer_config.json` available locally: `padding_side`
is unset, so that model would get `"right"`.)

**Why right padding is wrong here.** It matters for `activation_mode="generation"`
and `activation_mode="both"`, the two modes that call `model.generate`
(`src/taxonomy/functional.py:242`), for two separate reasons:

- *The model generates from the wrong position.* A decoder-only model continues from
  the **last position** of its input. Under right padding, any sequence shorter than
  the batch maximum ends in pad tokens, so the model is asked to continue
  `"…real prompt text <pad><pad>"` rather than the real end of the prompt.
  Transformers prints an explicit warning when it detects this.
- *The activation reader picks up a pad token's state.*
  `_extract_generation_activations` takes the last position at every decoding step
  (`src/taxonomy/functional.py:255`):

  ```python
  h = step_hs[layer_idx][:, -1, :]   # (batch, d)
  ```

  At step 0 the hidden states span the whole prompt, so `[:, -1, :]` returns the
  state at the final *input* position — a pad position for every row shorter than
  the batch maximum. That vector is then averaged into the per-(query, layer) mean
  over steps.

Under left padding both disappear: pads sit in front, so the last position is always
the true final prompt token for every row in the batch. Note that with right padding
only the rows that happen to be exactly the batch-maximum length are unaffected, so
whether a given query is affected depends on which other queries share its batch.

Setting `padding_side = "left"` in `FunctionalTaxonomy` to match behavioral is the
obvious candidate, but confirm it rather than assume it — the default
`activation_mode="input"` does not call `generate` at all, and is a different
question (next item).

### 3. Pooling averages over padded positions — noted, magnitude unclear

In `activation_mode="input"`, `_extract_input_activations` slices one query's hidden
states and hands them to `_pool` (`src/taxonomy/functional.py:217`):

```python
h = out.hidden_states[layer_idx][query_idx]   # (seq_len, d)
vec = self._pool(h).float().cpu().numpy()
```

`seq_len` here is the **padded** length, and `_pool` with `pooling="mean"` averages
over `dim=0` (`src/taxonomy/functional.py:269`). The `attention_mask` that records
which positions are real is present in `inputs` but is never passed to `_pool`. So
the mean includes pad-token hidden states, and `pooling="last_token"` under right
padding returns a pad position outright.

**This is recorded for consideration, not as a decided fix — how much it actually
matters is worth discussing first.** Pad-token states are not arbitrary noise; the
queries in a given experiment may be similar enough in length that the contamination
is small; and the downstream `normalize_activations` step plus the Gram/CKA
construction may absorb much of what remains. The question is whether it moves any
distance we care about, not whether it is theoretically impure.

If it seems worth measuring, the cheap empirical check is to extract one model twice
— once at `batch_size=1`, once at `batch_size=8` — and compare the resulting
matrices. `batch_size=1` needs no padding at all, so any difference between the two
is exactly the padding contribution.

---

## The query set is not pinned

`make_queries` (`scripts/_utils.py:412`) produces the shared query list. It rebuilds
a `MixedDataset` from `{output_dir}/datasets/{name}.recipe.json` on every call rather
than reading through `SampledDatasetCache`, and it takes its seed from the matching
entry in the config's `datasets` list rather than from an extraction-level setting.

This interacts badly with cache keying. Both `FunctionalTaxonomy.config_dict` and
`BehavioralTaxonomy.config_dict` include the full `queries` list
(`src/taxonomy/functional.py:82`, `src/taxonomy/behavioral.py:77`), and that config
is hashed into the `DiskCache` key as described in open question 1. So **for these
two levels specifically**, any drift in the recipe file or the seed changes every
key and silently invalidates the entire cache — no error, just a complete miss and a
full re-extraction.

`DatasetEmbeddingTaxonomy`, the third level in the same `extraction` section, is
unaffected: it does not use queries at all, which `scripts/extract_reprs.py:61-69`
already special-cases when deciding whether to load them.

The fix is already decided in `cache_layout_migration.md` (item 3 of "Three things
to know before starting"): keep the canonical query set in `01_datasets`, keyed by
`(recipe_hash, n_samples, seed)` like any other sampled dataset, plus a convenience
`queries.json` in `04_activations` / `05_generated` with that key recorded in
`config.json` so the canonical copy stays authoritative. Doing the cache renumbering
(item 1 in `TODO.md`, now done) therefore removed this problem rather than working
around it.

---

## Smallest run that would settle this

1. Pick 3-5 adapters that already exist in `results/shared_cache/03_adapters/`.
2. Run `python scripts/extract_reprs.py <config> --taxonomy functional behavioral`
   with a small `extraction.n_queries` (16-32 is enough to expose shape and padding
   questions).
3. Extend `scripts/check_analysis.py` with data-backed checks in the style of the
   existing `t_scan_cache` and `t_comparison_end_to_end` — those walk the real cache
   and report `SKIP` when a path is missing rather than passing vacuously, which is
   the property to copy. Worth covering:
   - representation shape matches what the configured `representation` mode promises,
     for both `gram` and `matrix`;
   - batch-invariance: same matrix at `batch_size=1` and `batch_size=8` (this is the
     check that answers item 3 above, and it is expected to fail before any fix);
   - `DiskCache` save → load round-trip preserves the matrix and the metadata;
   - `BehavioralTaxonomy` actually populates `metadata["generated_texts"]`
     (`src/taxonomy/behavioral.py:138`), since that is the only way to audit
     generations without re-running the model.
4. Only then wire the comparison layer, with open questions 1 and 2 answered.
