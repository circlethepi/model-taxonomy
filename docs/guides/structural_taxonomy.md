# Structural Taxonomy

The structural taxonomy compares models by the geometry of their **weight matrices**. No input data is required — the representation is derived directly from the model's parameters.

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

taxonomy = StructuralTaxonomy(
    layer_names=None,         # None = auto-detect (LoRA modules or all 2-D weights)
    n_components=256,         # per-layer vector length after truncate/pad
    lora_only=True,           # use LoRA adapter matrices only (default)
    use_lora_product=False,   # if True, compare B@A product instead of concat(A, B)
    cache=DiskCache("./cache"),
    hf_token=None,            # falls back to HF_TOKEN env var
)
```

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

By default (`use_lora_product=False`), the per-layer vector is:

```
v = concat(lora_A.flatten(), lora_B.flatten())
```

With `use_lora_product=True`, the actual weight delta is computed instead:

```
v = (lora_B @ lora_A).flatten()    # shape: (out_features * in_features,)
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

The named parameters must exist in the model. This mode is useful when you want to compare a specific subset of layers (e.g., only attention projections, or only the first few layers).

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

Choosing `n_components`:

| Scenario | Recommendation |
|---|---|
| Comparing LoRA adapters (rank 4–16) | 64–256 (the adapter vectors are naturally small) |
| Comparing attention weight matrices (4096×4096) | 512–2048 (more values capture more structure) |
| Quick comparison / diagnostic | 256 (default) |

Increasing `n_components` captures more of the weight structure but makes representations larger and distances slower to compute. The CKA metric is invariant to the specific value chosen as long as `n_components` is consistent across all models in a collection.

---

## Representation shape

```
(N_layers, n_components)
```

where `N_layers` is the number of LoRA modules (in LoRA mode) or specified/detected weight layers (in full-weight mode), and `n_components` is fixed.

---

## Distance metrics

Both `FrobeniusDistanceMetric` and `CKADistanceMetric` work with structural representations.

```python
from src import FrobeniusDistanceMetric, CKADistanceMetric

# Direct comparison of weight vector geometry
metric = FrobeniusDistanceMetric(normalize=True)

# Invariant to orthogonal transformations in weight space
metric = CKADistanceMetric(kernel="linear", unbiased=True)
```

**Tip:** `CKADistanceMetric(unbiased=True)` requires `N_layers >= 4` for numerically stable estimates. With fewer layers, use `unbiased=False`.

---

## Full example

```python
import torch
from src import (
    StructuralTaxonomy, CKADistanceMetric,
    MDSGeometry, LocalBackend, DiskCache,
    ModelCollection, TaxonomyAnalyzer,
)

# Compare LoRA fine-tuned variants of the same base model
models = ModelCollection.from_ids([
    "base-org/Llama-3.2-1B-lora-task-A",
    "base-org/Llama-3.2-1B-lora-task-B",
    "base-org/Llama-3.2-1B-lora-task-C",
])

# LoRA mode (default): use only adapter matrices
taxonomy = StructuralTaxonomy(
    lora_only=True,
    n_components=256,
    cache=DiskCache("./cache"),
    hf_token="hf_...",
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=CKADistanceMetric(kernel="linear", unbiased=False),
    geometry_method=MDSGeometry(n_components=2),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))

print(result.distance_matrix.sorted_neighbors("base-org/Llama-3.2-1B-lora-task-A"))
result.save("./results/structural_lora_cka")
```

### Full-weight example

```python
# Compare base models by specific attention layers
taxonomy = StructuralTaxonomy(
    lora_only=False,
    layer_names=[
        f"model.layers.{i}.self_attn.q_proj.weight"
        for i in range(4)   # first 4 layers
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
