# Structural Taxonomy

The structural taxonomy compares models by the geometry of their **weight matrices**. No input data or inference is required — the representation is derived directly from the model's parameters.

The default mode (`lora_only=True`) uses only LoRA adapter matrices, making this taxonomy practical for comparing fine-tuned variants of the same base model without storing full weight matrices.

## How it works

1. Obtain the adapter tensors — from `LoRACache`, from a local PEFT
   `adapter_model.safetensors`, or by loading the model on CPU as a last resort
   (no GPU needed; no inference required).
2. Identify the set of weight layers to compare (LoRA adapters or full weight matrices).
3. For each layer, construct a vector:
   - **LoRA mode (`use_lora_product=True`, default)**: `(B @ A).flatten()`.
   - **LoRA mode (`use_lora_product=False`)**: `concat(A.flatten(), B.flatten())`.
   - **Full-weight mode**: the flattened weight matrix.
4. Stack vectors across layers, zero-padding shorter rows to the longest:
   representation matrix of shape `(N_layers, max_len)`.
5. Unload the model from memory.

Vectors are kept at their full natural length — nothing is truncated, so no
weight information is discarded. Padding is only there to make the rows
stackable; the true pre-padding lengths are recorded in
`metadata["layer_lengths"]` and in the cache `config.json`. Rows differ in length
whenever the selected projections do (under GQA, `k_proj`/`v_proj` are narrower
than `q_proj`/`o_proj`).

## Configuration

```python
from src import StructuralTaxonomy
from src.cache import LoRACache

taxonomy = StructuralTaxonomy(
    layer_names=None,              # explicit module-name prefixes; overrides the shorthands below
    layer_indices="last",          # int, list[int], "last", or None for all layers
    projections=["q", "o"],        # "k"/"q"/"v"/"o" (or long forms), a list, or None for all
    lora_only=True,                # use LoRA adapter matrices only (default)
    use_lora_product=True,         # True = compare B@A product; False = concat(A, B)
    lora_cache=LoRACache("./cache"),  # hierarchical cache (recommended for LoRA)
    base_model_id=None,            # auto-detected from PEFT adapter_config.json
    cache=None,                    # flat DiskCache fallback (optional)
    hf_token=None,                 # falls back to HF_TOKEN env var
)
```

`layer_indices` and `projections` are architecture-agnostic shorthands matching
the conventions of `load_lora_weights`. Pass `layer_names` when you need explicit
control; it takes precedence over both.

Cache priority: `lora_cache` is checked first; `cache` (flat `DiskCache`) is used as a fallback if set.

---

## LoRA mode (default)

When `lora_only=True` (the default), only LoRA adapter matrices are used. This is the recommended setting when comparing fine-tuned variants of a single base model.

### Why LoRA only?

A LoRA-adapted layer adds two small matrices:
- `lora_A`: shape `(rank, in_features)`
- `lora_B`: shape `(out_features, rank)`

For `rank=16`, `in_features=out_features=4096`:

| Representation | Size |
|---|---|
| Full weight matrix | 16 777 216 values |
| LoRA A + B (concatenated) | 131 072 values |
| LoRA product `B @ A` | 16 777 216 values |

The LoRA matrices encode the *delta* applied to the base model during fine-tuning. Comparing these deltas directly captures what changed during fine-tuning, independent of the shared base weights.

### LoRA detection

LoRA adapters are detected automatically by scanning for parameters whose names contain `.lora_A.` and `.lora_B.`. This covers the standard PEFT naming convention:

```
model.layers.0.self_attn.q_proj.lora_A.default.weight
model.layers.0.self_attn.q_proj.lora_B.default.weight
```

If `layer_names` is provided alongside `lora_only=True`, only LoRA modules whose names start with one of the given prefixes are included.

### Vector construction

By default (`use_lora_product=True`), the actual weight delta is computed:

```
v = (lora_B @ lora_A).flatten()    # shape: (out_features * in_features,)
```

With `use_lora_product=False`, the raw adapter matrices are concatenated instead:

```
v = concat(lora_A.flatten(), lora_B.flatten())
```

The product `B @ A` represents the direct change to the weight matrix, but it is far larger than the concatenated factors — `out_features x in_features` rather than `rank x (in + out)`. Selecting fewer layers or projections is the way to keep representations small; for pairwise distances specifically, prefer the low-rank builders described under [Distance metrics](#distance-metrics), which compute the same numbers without ever materialising the product.

### When `lora_only=True` fails

If the model has no LoRA adapter parameters, `extract()` raises:

```
ValueError: lora_only=True but the model has no LoRA adapter parameters.
Use lora_only=False to compare full weight matrices instead.
```

This happens for base models (not fine-tuned with LoRA) and for models where LoRA weights were already merged into the base weights before saving. For merged models, use `lora_only=False`.

---

## LoRA cache (`LoRACache`)

`LoRACache` organises structural representations under a `base_model → adapter` hierarchy on disk, alongside a human-readable `config.json` per adapter.

> **HuggingFace compatibility note:** `LoRACache` is a *custom* cache for extracted representations. It sits alongside HuggingFace's own download cache (`~/.cache/huggingface/hub/`) and does not conflict with it. Raw LoRA weight tensors are not stored here — they stay in HuggingFace's cache. What `LoRACache` stores is the extracted representation matrix that would otherwise have to be recomputed each run.

### Directory structure

Locally trained adapters (produced by `scripts/finetune_lora.py`) are written to:

```
{output_dir}/adapters/
  meta-llama--Llama-3.1-8B/
    my_dataset_s42_r8_i00/    ← _s{seed} = data seed · _r{rank} = LoRA rank · _i{seed:02d} = init seed
      adapter_config.json
      adapter_model.safetensors
```

`LoRACache` stores extracted representations alongside a `config.json` under a parallel hierarchy:

```
cache_root/adapters/
  meta-llama--Llama-3.1-8B/            ← base model (/ replaced with --)
    some-org--my-adapter/               ← adapter (/ replaced with --)
      config.json
      representation.safetensors
```

### `config.json` schema

```json
{
  "schema_version": "1",
  "base_model_id": "meta-llama/Llama-3.1-8B",
  "adapter_id": "some-org/my-adapter",
  "adapter_type": "lora",
  "training_config": {
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_init_seed": 0,
    "target_modules": ["q_proj", "v_proj"],
    "lora_dropout": 0.05
  },
  "dataset_recipe": {
    "_note": "stub — populate with actual dataset details",
    "dataset_ids": [],
    "split": null,
    "num_samples": null
  },
  "extraction_config": {
    "layer_names": null,
    "layer_indices": [27],
    "projections": ["o"],
    "lora_only": true,
    "use_lora_product": true
  },
  "layer_lengths": [9437184],
  "extracted_at": "2026-06-15T00:00:00Z"
}
```

`extraction_config` is what the `{config_hash}` directory name is derived from, so
two different layer/projection selections for the same adapter are cached side by
side rather than overwriting each other. `layer_lengths` records each row's true
length before zero-padding.

`training_config` is populated automatically from the adapter's PEFT `adapter_config.json` (downloaded from the Hub). `dataset_recipe` can be passed explicitly to `LoRACache.save()` to record the full mixing recipe; if omitted, a placeholder stub is written instead.

`lora_init_seed` records the random seed passed to `torch.manual_seed()` immediately before PEFT initialises the A and B matrices. Holding all other hyperparameters fixed while varying `lora_init_seed` produces a distinct adapter directory (`_i{seed:02d}` suffix), which makes seed-sensitivity studies straightforward to run and cache independently.

### Base model auto-detection

When `base_model_id=None` (the default), `StructuralTaxonomy` reads the adapter's `adapter_config.json` from the Hub to find `base_model_name_or_path`. You can also specify it explicitly:

```python
taxonomy = StructuralTaxonomy(
    lora_cache=LoRACache("./cache"),
    base_model_id="meta-llama/Llama-3.1-8B",   # skip the Hub lookup
)
```

### Cache API

```python
from src.cache import LoRACache

lc = LoRACache("./cache")
base, adapter = "meta-llama/Llama-3.1-8B", "some-org/my-adapter"

# Which representation to address — the same dict StructuralTaxonomy uses to
# build the {config_hash} directory name.
cfg = {
    "layer_names": None, "layer_indices": [27], "projections": ["o"],
    "lora_only": True, "use_lora_product": True,
}

# Check and load
lc.exists(base, adapter, cfg)       # → bool
lc.load(base, adapter, cfg)         # → ModelRepresentation
lc.load_config(base, adapter, cfg)  # → dict (config.json)

# Browse
lc.list_base_models()                    # → ["meta-llama/Llama-3.1-8B", ...]
lc.list_adapters(base)                   # → adapters with an extracted representation
lc.list_raw_adapters(base)               # → adapters with raw PEFT files present
lc.adapter_status(base)                  # → {"processed": [...], "raw": [...]}
lc.adapter_count(base)                   # → {"processed": N, "raw": M}
lc.list_representations(base, adapter)   # → [(config_hash, extraction_config), ...]
```

---

## Full-weight mode (`lora_only=False`)

When `lora_only=False`, full weight matrices are used.

### With explicit `layer_names`

```python
taxonomy = StructuralTaxonomy(
    lora_only=False,
    layer_names=[
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.1.self_attn.q_proj.weight",
        "model.layers.1.self_attn.v_proj.weight",
    ],
)
```

To find available parameter names for a model:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("model-id", device_map="cpu")
for name, param in model.named_parameters():
    if param.ndim == 2:
        print(name, param.shape)
del model
```

### With automatic layer selection (`layer_names=None`)

When `layer_names=None` and `lora_only=False`, all 2-D weight matrices with at least 1 024 elements are included automatically. This typically captures all attention and MLP projection matrices.

```python
taxonomy = StructuralTaxonomy(lora_only=False)
```

**Caution:** Different architectures name their layers differently. Comparing models across architectures in this mode will generally produce different `N_layers` values, which will fail shape validation in `TaxonomyAnalyzer`. The structural taxonomy is most meaningful when comparing models that share the same architecture.

---

## Controlling representation size

There is no truncation parameter. Every vector is stored at full length, so the
only way to control size is to select fewer blocks — via `layer_indices` and
`projections` (or explicit `layer_names`).

| Scenario | Recommendation |
|---|---|
| Quick diagnostic | `layer_indices="last"`, `projections="o"` |
| Depth sweep | `layer_indices=[0, 13, 27]`, one projection |
| Full comparison | `layer_indices=None`, `projections=None` — but prefer the low-rank distance builders below |

---

## Distance metrics

### Through the pipeline

```python
from src import FrobeniusDistanceMetric, CKADistanceMetric, CosineDistanceMetric

# Angle between the concatenated weight-delta vectors
metric = CosineDistanceMetric()

# Direct comparison of weight vector geometry
metric = FrobeniusDistanceMetric(normalize=True)

# Invariant to orthogonal transformations in weight space
metric = CKADistanceMetric(kernel="linear", unbiased=True)
```

`CKADistanceMetric(unbiased=True)` requires `N_layers >= 4`. With fewer layers, use `unbiased=False`.

`CosineDistanceMetric` flattens the whole representation before comparing. Since
zeros contribute nothing to either a dot product or a norm, the zero padding
described above is inert, and a consistent reordering of entries across both
vectors leaves cosine unchanged — so it is well defined on structural
representations exactly as stored. Select it from an experiment YAML with
`metrics: {structural: cosine}`.

> `FrobeniusDistanceMetric` and `CKADistanceMetric` both carry a
> `TODO (full-pipeline)` noting they assume uniform-length rows and have not yet
> been adapted to zero-padded structural representations. Cosine is unaffected.

### Directly from LoRA factors (recommended)

For pairwise distances there is no need to materialise `B @ A` at all. The
builders in `src.notebook.structure` work entirely in rank space, and
`src.analysis.lora_distance_matrix` wraps them so the result is an ordinary
`DistanceMatrix`:

```python
from src.notebook.lora_weights import load_lora_weights
from src.analysis import lora_distance_matrix, fit_geometry

weights = load_lora_weights(adapter_names, adapter_root="results/shared_cache/adapters",
                            layer_indices=list(range(28)), projections=["k", "q", "v", "o"])

dm = lora_distance_matrix(weights, kind="cosine")     # or "frobenius", "bures_wasserstein", "cka"
geo = fit_geometry(dm, method="mds", n_components=2)
```

`kind="cosine"` returns exactly what `CosineDistanceMetric` would, without ever
forming a `d x d` matrix — the only tractable route at 28 layers x 4 projections
x 3072². See `docs/notes/frobenius_bw_generalization.md` for why summing across
blocks is exact, and `docs/notes/cka_notes.md` for why `kind="cka"` is restricted
to a single block.

---

## Full example

```python
import torch
from src import (
    StructuralTaxonomy, CKADistanceMetric,
    MDSGeometry, LocalBackend,
    ModelCollection, TaxonomyAnalyzer,
)
from src.cache import LoRACache

# Compare LoRA fine-tuned variants of the same base model
models = ModelCollection.from_ids([
    "some-org/Llama-3.1-8B-lora-task-A",
    "some-org/Llama-3.1-8B-lora-task-B",
    "some-org/Llama-3.1-8B-lora-task-C",
])

taxonomy = StructuralTaxonomy(
    lora_only=True,
    layer_indices="last",
    projections="o",
    lora_cache=LoRACache("./cache"),
    # base_model_id auto-detected from each adapter's adapter_config.json
    hf_token="hf_...",
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=CKADistanceMetric(kernel="linear", unbiased=False),
    geometry_method=MDSGeometry(n_components=2),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))

print(result.distance_matrix.sorted_neighbors("some-org/Llama-3.1-8B-lora-task-A"))
result.save("./results/structural_lora_cka")

# Inspect the cached config for any adapter
lc = LoRACache("./cache")
cfg = lc.load_config(
    "meta-llama/Llama-3.1-8B",
    "some-org/Llama-3.1-8B-lora-task-A",
    taxonomy._extraction_config(),
)
print(cfg["training_config"])
print(cfg["dataset_recipe"])   # stub, to be filled in
```

### Full-weight example

```python
# Compare base models by specific attention layers
from src.cache import DiskCache

taxonomy = StructuralTaxonomy(
    lora_only=False,
    layer_names=[
        f"model.layers.{i}.self_attn.q_proj.weight"
        for i in range(4)
    ],
    cache=DiskCache("./cache"),
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=FrobeniusDistanceMetric(normalize=True),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))
```
