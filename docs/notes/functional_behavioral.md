# Functional and behavioral taxonomies: details still to work out

> **Status update (2026-08-03) — most of this note is now historical.**
>
> Both levels have run. Behavioral landed first (`05_generated/`,
> `GeneratedTextCache`, `behavioral_repr`, `_behavioral_matrix`); functional
> followed (`04_activations/`, `ActivationCache`, `functional_repr`,
> `_functional_matrix`).
>
> **Open questions 1, 2 and 3 are closed** — see the resolution box at the head of
> each. **Question 4 is still open** and is item 8 in [TODO.md](TODO.md).
>
> Of "things to check": item 1 (embedders pinned to CPU) was fixed; item 2
> (functional does not pin padding side) is fixed — the pin now lives on the
> shared `HFInferenceTaxonomy` base class; item 3 (pooling over padded positions)
> is fixed and **measured** rather than assumed.
>
> The `gram` representation was **redefined** as part of this work: rows are
> queries, not layers. That is documented in [gram_and_cka.md](gram_and_cka.md),
> which is the note to read on Gram matrices and CKA.
>
> The original text is kept below because the reasoning behind each decision is
> still the record of *why* things are the way they are.

**Original status:** both levels are fully implemented and fully wired into the
scripts, and neither has ever been run. Nothing below is a bug report — these are
the details to settle and the things to look at when the first extraction happens.

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

**Greedy decoding is not reproducible across batch sizes, and that is not a bug.**
Worth stating because two versions of the batch-invariance check assumed otherwise.
Batched matmuls tile differently, so fp16 logits differ in their last bits, and
greedy `argmax` flips wherever two tokens are near-tied; the sequences then diverge
and never reconverge. Measured on an L40S (job 1987293, 8 queries, batch 1 vs 8):
**6/8 byte-identical**, the 2 divergent ones splitting ~10 % in after ~50 characters
of shared prefix into two equally fluent continuations, with **no correlation to
padding amount** — the shortest prompt (most padding) was identical, the divergent
ones were mid-length.

Left padding is therefore correct. The distinguishing signature, which
`t_behavioral_batch_invariance` now asserts instead of equality:

| | fp16 tie-flipping | broken padding |
|---|---|---|
| how many diverge | a minority | most of the batch |
| where they diverge | mid-sequence, after a shared prefix | from the first generated token |
| correlation with padding amount | none | shortest prompts worst |

Practical consequence: a behavioral representation is reproducible only at a fixed
batch size, and `batch_size` is deliberately **not** in `config_dict()`, so the cache
will not distinguish two runs that used different ones. `metadata` records
`batch_size` and `device_name` for exactly this reason. Distances are affected only
to the extent that a minority of generations differ, and the observed per-row cosine
between batch-1 and batch-8 vectors was 1.000 for matching rows and 0.84–0.87 for the
two that flipped.

**3. Test coverage: behavioral now, functional still not.**
`scripts/check_analysis.py` is the repo's verification script and its only test
harness — there is no pytest. It registers checks in three tiers: **37 synthetic,
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

> **CLOSED.** The premise dissolved rather than the question being answered:
> neither level uses `DiskCache` any more, so neither is config-keyed in the way
> described below. Behavioral moved to `GeneratedTextCache`
> (`05_generated/{config_hash}/…`) and functional to `ActivationCache`
> (`04_activations/{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/…`). Both make
> availability a **file-existence test** again — option (a)'s uniformity without
> its walk-and-read cost.
>
> Both tokens carry the **same honesty caveat**, documented on the `scan_cache`
> arguments: given `behavioral_config_hash` / `functional_draw` the token is
> exact; without one it degrades to "some representation exists, under some
> config/draw", which is weaker than it looks, because a representation from a
> different query set is not interchangeable. Pass the selector whenever the
> answer will choose models for a comparison.
>
> Both checks ask the cache where its own files live rather than rebuilding the
> path — `_sampled_rows_exist` is the counterexample still in the tree.

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

> **CLOSED — and the `gram` option below was redefined, not merely chosen between.**
>
> The `representation` parameter is gone. Extraction now stores **pooled per-layer
> activations**, one `(n_queries, d)` file per `(mode, pooling, layer)`, and the
> matrix handed to the comparison layer is a **view** assembled at read time. That
> removes the "which mode do we extract in?" question entirely: both views come
> from one run, and switching costs a surrogate build rather than a GPU pass.
>
> | view | shape | rows are |
> |---|---|---|
> | `concat` (**default**, feeds the metrics) | `(n_queries, L·d)` | queries |
> | `gram` | `(n_queries, n_queries)` | queries |
>
> The old `gram` — stacked per-layer triangles, rows = **layers** — was dropped
> rather than kept as a second option. Two objects sharing a name with different
> row semantics is precisely the confusion worth not shipping, and nothing had
> been computed with it. `gram` now means `G = H Hᵀ` of the concatenation.
>
> The observation below that the two modes are "not interchangeable" was right,
> and is why the redefinition matters: rows must be **queries** for a
> query-set CKA to mean anything.
>
> One trap this creates: a stored Gram is a *kernel*, and `CKADistanceMetric`
> forms its own, so passing one computes `(H Hᵀ)²` silently. The cache tags kernel
> views and the metric refuses them. See [gram_and_cka.md](gram_and_cka.md).

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

> **PARTLY CLOSED — half of this is fixed, and the half that remains is the
> interesting half.**
>
> **Fixed:** functional no longer stores anything under a configured (possibly
> negative) index. `_resolve_layers` resolves against the model's actual
> `n_hidden_states` at extraction, and `ActivationCache.activation_path` **raises**
> on a negative index. `-1` and `28` can no longer become two files that drift
> apart, and `runs/{hash}.json` records both the resolved layers and
> `n_hidden_states`, so the numbering used is always recoverable from disk.
>
> **Still open:** the structural↔functional mapping. Functional indexes
> `hidden_states` (length `n_layers + 1`, index 0 = embedding output); structural
> uses absolute adapter layer numbers. The offset is knowable — functional index
> `k+1` is the output of transformer block `k` — but nothing in the code states
> it, so a cross-level claim of "the same layer in both" is still unverified.
> Cheap to add once someone wants to make that claim.

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

> **FIXED.** The pin now lives on `HFInferenceTaxonomy`, a base class both levels
> inherit, so it cannot be set for one and forgotten for the other — which is the
> shape of bug this item actually was. `scripts/check_analysis.py` asserts the pin
> is present in the source for **both** taxonomies, via `inspect.getsource`, so it
> fails the moment the line is removed.
>
> The docstring records the reason it matters for functional specifically, which
> is different from behavioral's: with left padding the last real token is always
> at index `-1`, so `last_token` pooling finds it without mask arithmetic.

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

> **FIXED, and the magnitude is now measured rather than assumed.**
>
> `_pool` takes the `attention_mask` and pools only over real positions:
> `mean` divides by the unpadded length, `last_token` indexes the last unmasked
> position, `cls` the first. The property this buys is that **a pooled vector is a
> function of its query alone** — it no longer depends on which other queries
> shared its batch, which matters because `batch_size` is deliberately not part of
> the cache key, so two runs at different batch sizes share an entry.
>
> Two checks pin it, at different costs:
> - *synthetic* — hand-built tensors with known padding; asserts padded and
>   unpadded pooling agree for all three modes, **and** that the unmasked mean
>   provably differs, so the check cannot pass vacuously. No GPU, milliseconds.
> - *`[gpu]`* — `batch_size=1` (no padding at all) against one full batch (maximum
>   padding), asserting per-row cosine > 0.999. This is the measurement this item
>   asked for: the gap between the two arms *is* the padding contribution. It can
>   be much tighter than the behavioral equivalent because `input` mode runs no
>   `generate` call, so there is no greedy argmax to flip on a near-tie — the only
>   residual is fp16 matmul tiling.
>
> See the "Verification" section of [TODO.md](TODO.md) for the measured numbers
> once the smoke run has been read.

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
