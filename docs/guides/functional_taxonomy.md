# Functional Taxonomy

The functional taxonomy compares models by their **internal activations** over a
shared set of query inputs. Rather than looking at what a model produces
(behavioral taxonomy) or what its weights look like (structural taxonomy), it
asks: *how similarly do two models represent the same inputs internally?*

## The central idea: extraction stores, reading selects

One forward pass produces **every** hidden state at once. So "which layers am I
comparing?" is a question about *reading*, not about running a model.

- **Extraction** pools each layer's hidden states to one vector per query and
  stores `(n_queries, d)` — one file per layer — in
  [`ActivationCache`](../api_reference.md).
- **Reading** assembles a *view* from whichever layers you want, at read time.

By default extraction stores all layers (~23 MB for a 3B model at 64 queries) and
reading concatenates across all of them. Changing your mind about layers costs a
surrogate build, not a GPU pass.

## Views

| View | Definition | Shape | Rows are |
|---|---|---|---|
| `"concat"` (default) | Activations concatenated across the selected layers | `(n_queries, L·d)` | queries |
| `"gram"` | `G = H Hᵀ` of that concatenation | `(n_queries, n_queries)` | queries |

**Rows are queries in both.** An earlier version of this taxonomy stacked
per-layer Gram triangles, making a row a *layer*; that form is gone. See
[`notes/gram_and_cka.md`](../notes/gram_and_cka.md) for why, and for the trap
that comes with `gram`.

> **`gram` is a kernel, not a feature matrix.** `CKADistanceMetric` forms
> `K = X Xᵀ` itself, so handing it a stored Gram computes `(H Hᵀ)²` — a different
> quantity that still returns a plausible number. Feed CKA the `concat` view;
> linear CKA forms exactly this Gram internally anyway. The cache tags kernel
> views and the metric refuses them, so this fails loudly rather than silently.

Views are computed lazily and **written back** under `surrogates/{hash}/`, so a
given `(draw, mode, pooling, layers, view, normalize)` is computed at most once.

## Activation modes

| Mode | What is captured | Stored as |
|---|---|---|
| `"input"` (default) | Forward pass on the prompt | `input_{pooling}_layer{NNN}.safetensors` |
| `"generation"` | Last-token hidden state at each decoding step, mean-pooled | `generation{max_new_tokens}_{pooling}_layer{NNN}.safetensors` |
| `"both"` | Both, combined along the feature axis at read time | both files above |

There is no stored `both`: it is a read-time combination. That is what lets a
draw that already has `input` gain `generation` later without recomputing either.
Generation carries its token budget in the filename because 32 tokens and 128
tokens produce different vectors.

## Pooling is mask-aware

Pooling ignores padded positions. This is what makes a pooled vector a function
of **its query alone**.

`padding=True` pads each batch to *its own* longest sequence, so an unmasked mean
would average in pad-position hidden states — which are not zero, since the model
computes a residual-stream vector at every position even though nothing attends
to them — and how many a query gets would depend on which other queries shared
its batch. Reordering queries or changing `batch_size` would then shift every
vector, and the cache could not notice, because neither is part of the key.

| Option | Description |
|---|---|
| `"mean"` | Average over unpadded token positions (default) |
| `"last_token"` | The last *unmasked* position; natural for causal LMs |
| `"cls"` | The first unmasked position; only meaningful for BERT-style models |

`scripts/check_analysis.py` pins this at two tiers: a synthetic check on
hand-built tensors, and a `[gpu]` check comparing `batch_size=1` (no padding at
all) against one full batch (maximum padding).

## Normalization is a read-time property

`normalize_activations` applies when a view is assembled, not when activations
are stored. Raw activations go to disk; normalization is part of a surrogate's
identity. A normalized and an unnormalized view of the same run coexist without
re-running inference.

When on (the default), each row of the concatenation is divided by its L2 norm,
which makes `G[i,i] = 1` in the `gram` view and makes distances depend on the
direction of activations rather than their magnitude.

## `layer_indices`

Indices into the `hidden_states` tuple: index 0 is the embedding layer, then one
per transformer block.

**`None` (the default) stores every layer** — recommended, since restricting buys
no GPU time. Negative indices are resolved to absolute positions before anything
touches disk, so `-1` and `28` cannot be stored twice under two names and drift
apart.

## Configuration

```python
from src.taxonomy.functional import FunctionalTaxonomy
from src.cache.activation_cache import ActivationCache

taxonomy = FunctionalTaxonomy(
    queries=queries,
    query_key={"recipe_hash": "...", "n_samples": 64, "seed": 0},
    layer_indices=None,                   # None = store every hidden state
    cache=ActivationCache("results/shared_cache"),
    device="cuda",
    batch_size=16,
    torch_dtype=torch.float16,
    pooling="mean",
    normalize_activations=True,           # read-time; part of the surrogate key
    activation_mode="input",
    max_new_tokens=32,                    # used when mode != "input"
    view="concat",
)
```

`query_key` — the `{recipe_hash, n_samples, seed}` triple identifying the draw in
`01_datasets` — is required, and is what keys the cache. The query *strings* are
never hashed: they are derived data, and hashing them would make every entry
sensitive to any upstream change that shifts the draw, with no way to tell from a
key which draw an entry belonged to.

An `ActivationCache` is required. Per-layer activations are the stored artefact
and views are assembled from them; there is no in-memory-only path.

The base model is loaded **once** and adapters are swapped onto it, so call
`close()` — or use the taxonomy as a context manager — when done.

## Storage layout

```
04_activations/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}/
    queries.json                              ← query_key + source row indices
    runs/{config_hash}.json                   ← provenance: resolved layers, batch, device
    activations/{mode}_{pooling}_layer{NNN}.safetensors
    surrogates/{surrogate_hash}/config.json + surrogate.safetensors
```

Keyed **model-wise then draw-wise**, matching `LoRACache`, so a functional entry
sits at the same `{base}/{adapter}` coordinates as the structural entry for the
same adapter. Writes are purely additive: one file per `(mode, pooling, layer)`
means a later run that adds a mode never rewrites what is already there.

`queries.json` stores source row *indices*, not text — `01_datasets` is canonical
and `(recipe_hash, n_samples, seed)` determines the strings completely.

## Distance metrics

`CKADistanceMetric` is the default for this level. It is invariant to orthogonal
transformations and isotropic scaling, which matters when comparing
representations across models.

```python
from src.metrics.cka import CKADistanceMetric
metric = CKADistanceMetric(kernel="linear")     # unbiased=True by default
```

The unbiased HSIC estimator divides by `n(n-3)`, so it needs **at least 4 rows**
and raises below that rather than returning NaN into a distance matrix. Rows are
queries, so any realistic query set clears this easily. Pass `unbiased=False` for
small row counts. `FrobeniusDistanceMetric` also works on the `concat` view.

## Query set design

- Use diverse queries covering the input distribution of interest. Near-identical
  queries produce a near-rank-1 Gram and near-zero distances.
- 50–200 is a reasonable range. Storage is `n_queries × d` floats per layer:
  64 queries × 3072 dims ≈ 786 KB per layer, ~23 MB for all 29 layers of a 3B
  model.
- Use the **same draw** across levels when comparing them. `experiments/
  yahoo_functional_smoke.yaml` and `yahoo_behavioral_smoke.yaml` share a
  byte-identical `datasets:` block for exactly this reason.

## Running it

```bash
python scripts/extract_reprs.py experiments/yahoo_functional_smoke.yaml --taxonomy functional
```

Then read it back through the comparison layer:

```python
from src.analysis import build_taxonomy_artifacts, scan_cache

index = scan_cache("results/shared_cache", functional_draw=draw)
dm, geoms = build_taxonomy_artifacts(
    index.with_available("functional_repr"), "functional", metric="cka",
    functional_selector={"draw": draw, "mode": "input", "view": "concat"},
)
```

Every field of `functional_selector` is optional; `draw=None` resolves to the one
draw present when there is exactly one, and refuses when several are, since
different draws are different query sets and are not comparable.
