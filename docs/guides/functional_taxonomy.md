# Functional Taxonomy

The functional taxonomy compares models by the **covariance structure of their internal activations** across a shared set of probe inputs. Rather than looking at what a model produces (behavioral taxonomy) or what its weights look like (structural taxonomy), this taxonomy asks: *how similarly do two models process inputs internally, layer by layer?*

## How it works

For each model and each specified layer L:

1. Load the model and tokenizer from HuggingFace.
2. For each probe string (processed in batches):
   - Run a forward pass with `output_hidden_states=True`.
   - Extract the hidden states at layer L: `(batch, seq_len, d)`.
   - Pool over the sequence dimension to get one vector per probe: `h_i ∈ R^d`.
3. Stack all probe vectors: `H_L = (N_probes, d)`.
4. Optionally L2-normalize each row (`normalize_activations=True`, the default). This makes the Gram matrix a correlation matrix with unit diagonal.
5. Compute the Gram matrix: `G_L = H_L @ H_L.T → (N_probes, N_probes)`.
   - Entry `G_L[i, j]` is the dot product of the activation of probe `i` with the activation of probe `j` at layer L.
6. Flatten the upper triangle (including diagonal): `N_probes*(N_probes+1)//2` values per layer.
7. Stack across all N_layers: final representation matrix of shape `(N_layers, N_probes*(N_probes+1)//2)`.
8. Delete the model from GPU memory; clear the CUDA cache.

The representation matrix captures the *covariance geometry* of the activation space at each layer. Two models are similar under this taxonomy if their probe activations have the same pairwise similarity structure, layer by layer.

## Configuration

```python
from src import FunctionalTaxonomy

taxonomy = FunctionalTaxonomy(
    probes=probes,                        # shared input strings
    layer_indices=[-4, -3, -2, -1],       # which transformer layers to analyze
    cache=DiskCache("./cache"),           # recommended
    device="cuda",
    batch_size=8,
    torch_dtype=torch.bfloat16,           # use bfloat16 for Llama/Gemma
    hf_token=None,                        # falls back to HF_TOKEN env var
    pooling="mean",                       # how to reduce seq_len dimension
    normalize_activations=True,           # L2-normalize before computing Gram
)
```

### `layer_indices`

Indices into the `hidden_states` tuple returned by the model, which includes the embedding layer at index 0 followed by each transformer block:

| Index | Meaning |
|---|---|
| `0` | Embedding layer output (before any transformer blocks) |
| `1` | Output of first transformer block |
| `-1` | Output of last transformer block |
| `[-4, -3, -2, -1]` | Last 4 transformer blocks |

For comparing models *architecturally*, use consecutive indices from the last few layers (e.g., `[-4, -3, -2, -1]`). For a quick single-layer comparison, `[-1]` is a reasonable default.

### `pooling`

Controls how the `(seq_len, d)` hidden state tensor is reduced to a single `(d,)` vector per probe:

| Option | Description |
|---|---|
| `"mean"` | Average over all token positions (recommended, robust) |
| `"last_token"` | The final token's representation; useful for causal LMs |
| `"cls"` | The `[CLS]` token; only meaningful for BERT-style masked LMs |

### `normalize_activations`

When `True` (default), each row of `H_L` is divided by its L2 norm before computing the Gram matrix. This:
- Makes `G_L[i, i] = 1` for all probes (correlation matrix, not covariance).
- Removes scale differences between layers and models.
- Makes the distance metric depend only on the *direction* of activations, not their magnitude.

Set to `False` if you want to capture magnitude differences as well.

---

## Representation shape

The matrix stored in `ModelRepresentation` has shape:

```
(N_layers, N_probes * (N_probes + 1) // 2)
```

where `N_layers = len(layer_indices)` and `N_probes = len(probes)`.

Example: 4 layers, 50 probes → `(4, 1275)`.

---

## Distance metrics

Both existing metrics work with functional representations:

### `FrobeniusDistanceMetric`

Compares the flattened Gram vectors directly. Sensitive to exact values of all pairwise dot products.

```python
from src import FrobeniusDistanceMetric
metric = FrobeniusDistanceMetric(normalize=True)
```

### `CKADistanceMetric`

Computes CKA between the `(N_layers, ...)` matrices. Captures whether the *pattern* of inter-layer Gram structures is similar. More invariant to linear transformations than Frobenius.

```python
from src import CKADistanceMetric
metric = CKADistanceMetric(kernel="linear", unbiased=False)
```

**Important:** The unbiased HSIC estimator used by default in `CKADistanceMetric(unbiased=True)` requires the number of *rows* in the representation matrix (i.e., `N_layers`) to be at least 4. For smaller layer sets, use `unbiased=False`:

```python
# Safe for any N_layers >= 1:
metric = CKADistanceMetric(unbiased=False)

# Requires N_layers >= 4:
metric = CKADistanceMetric(unbiased=True)
```

---

## Layerwise comparison

The representation matrix is organized such that **each row corresponds to one layer**. This means:

- Models with the same architecture and the same `layer_indices` always produce matrices of the same shape — a prerequisite for computing distances.
- The distance metric operates on the full layerwise structure. `CKADistanceMetric` on two `(N_layers, d)` matrices asks: are the inter-layer covariance patterns similar?
- `FrobeniusDistanceMetric` sums differences across all layers and all Gram entries.

For comparing models across architectures (different numbers of layers), restrict to a common set of layers by their absolute index or by selecting specific layer names.

---

## Probe set design

The same guidelines as for the behavioral taxonomy apply:

- Use diverse probes that cover the input distribution of interest.
- Avoid too-similar probes — if all probes produce nearly identical activations, the Gram matrix will be near-rank-1 for all models and distances will be near zero.
- 50–200 probes is a reasonable range. Fewer probes reduce the size of the Gram matrix (beneficial for storage) but may underrepresent the activation geometry.

The Gram matrix has `N_probes * (N_probes + 1) / 2` entries per layer. For 100 probes and 4 layers, this is `100*101/2 * 4 = 20 200` float32 values per model — well within practical cache size.

---

## Full example

```python
import torch
from datasets import load_dataset
from src import (
    FunctionalTaxonomy, CKADistanceMetric,
    MDSGeometry, LocalBackend, DiskCache,
    ModelCollection, TaxonomyAnalyzer,
)

models = ModelCollection.from_ids([
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B",
])

# Use 50 MMLU questions as probes
ds = load_dataset("cais/mmlu", "all", split="test[:50]")
probes = [row["question"] for row in ds]

taxonomy = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-4, -3, -2, -1],     # last 4 layers
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=8,
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
    pooling="mean",
    normalize_activations=True,
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=CKADistanceMetric(kernel="linear", unbiased=False),
    geometry_method=MDSGeometry(n_components=2),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))

print(result.distance_matrix.sorted_neighbors("meta-llama/Llama-3.2-1B"))
print(result.geometry.coordinates)
result.save("./results/functional_cka")
```
