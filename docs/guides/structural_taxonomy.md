# Structural Taxonomy

The structural taxonomy compares models by the geometry of their **weight matrices**. No input data or inference is required — the representation is derived directly from the model's parameters.

The default mode (`lora_only=True`) uses only LoRA adapter matrices, making this taxonomy practical for comparing fine-tuned variants of the same base model without storing full weight matrices.

## How it works

1. Load the model on CPU (no GPU needed; no inference required).
2. Identify the set of weight layers to compare (LoRA adapters or full weight matrices).
3. For each layer, construct a vector:
   - **LoRA mode**: concatenate the flattened A and B adapter matrices, then truncate/pad to `n_components`.
   - **Full-weight mode**: flatten the weight matrix, then truncate/pad to `n_components`.
4. Stack vectors across layers: final representation matrix of shape `(N_layers, n_components)`.
5. Unload the model from memory.

## Configuration

```python
from src import StructuralTaxonomy
from src.cache import LoRACache

taxonomy = StructuralTaxonomy(
    layer_names=None,              # None = auto-detect (LoRA modules or all 2-D weights)
    n_components=256,              # per-layer vector length after truncate/pad
    lora_only=True,                # use LoRA adapter matrices only (default)
    use_lora_product=True,         # True = compare B@A product; False = concat(A, B)
    lora_cache=LoRACache("./cache"),  # hierarchical cache (recommended for LoRA)
    base_model_id=None,            # auto-detected from PEFT adapter_config.json
    cache=None,                    # flat DiskCache fallback (optional)
    hf_token=None,                 # falls back to HF_TOKEN env var
)
```

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
| After truncation to `n_components=256` | 256 values |

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

The product `B @ A` represents the direct change to the weight matrix, but it is much larger than the concatenated factors and may require a larger `n_components` to capture meaningful structure.

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

```
cache_root/loras/
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
    "n_components": 256,
    "use_lora_product": true,
    "layer_names": null
  },
  "extracted_at": "2026-06-15T00:00:00Z"
}
```

`training_config` is populated automatically from the adapter's PEFT `adapter_config.json` (downloaded from the Hub). `dataset_recipe` is a documented stub for the dataset taxonomy feature (to be implemented separately).

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

# Check and load
lc.exists("meta-llama/Llama-3.1-8B", "some-org/my-adapter")   # → bool
lc.load("meta-llama/Llama-3.1-8B", "some-org/my-adapter")     # → ModelRepresentation
lc.load_config("meta-llama/Llama-3.1-8B", "some-org/my-adapter")  # → dict (config.json)

# Browse
lc.list_base_models()                              # → ["meta-llama/Llama-3.1-8B", ...]
lc.list_adapters("meta-llama/Llama-3.1-8B")       # → ["some-org/my-adapter", ...]
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
    n_components=256,
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
taxonomy = StructuralTaxonomy(lora_only=False, n_components=256)
```

**Caution:** Different architectures name their layers differently. Comparing models across architectures in this mode will generally produce different `N_layers` values, which will fail shape validation in `TaxonomyAnalyzer`. The structural taxonomy is most meaningful when comparing models that share the same architecture.

---

## `n_components`

Each per-layer vector is truncated or zero-padded to exactly `n_components` values before stacking. This ensures the representation matrix has a fixed second dimension regardless of the actual layer sizes.

| Scenario | Recommendation |
|---|---|
| Comparing LoRA adapters (rank 4–16) | 64–256 (the adapter vectors are naturally small) |
| Comparing attention weight matrices (4096×4096) | 512–2048 |
| Quick comparison / diagnostic | 256 (default) |

---

## Distance metrics

```python
from src import FrobeniusDistanceMetric, CKADistanceMetric

# Direct comparison of weight vector geometry
metric = FrobeniusDistanceMetric(normalize=True)

# Invariant to orthogonal transformations in weight space
metric = CKADistanceMetric(kernel="linear", unbiased=True)
```

`CKADistanceMetric(unbiased=True)` requires `N_layers >= 4`. With fewer layers, use `unbiased=False`.

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
    n_components=256,
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
cfg = lc.load_config("meta-llama/Llama-3.1-8B", "some-org/Llama-3.1-8B-lora-task-A")
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
    n_components=512,
    cache=DiskCache("./cache"),
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=FrobeniusDistanceMetric(normalize=True),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))
```
