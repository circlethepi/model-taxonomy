# Functional Taxonomy

The functional taxonomy compares models by the **covariance structure of their internal activations** across a shared set of probe inputs. Rather than looking at what a model produces (behavioral taxonomy) or what its weights look like (structural taxonomy), this taxonomy asks: *how similarly do two models process inputs internally, layer by layer?*

## Activation modes

`FunctionalTaxonomy` supports three activation modes, controlled by the `activation_mode` parameter:

| Mode | What is captured | When to use |
|---|---|---|
| `"input"` (default) | Activations during the forward pass on the prompt | Comparing how models encode inputs |
| `"generation"` | Activations at each decoding step, mean-pooled | Comparing how models generate |
| `"both"` | Input and generation activations stacked together | Full picture of model processing |

## How it works

For each model and each specified layer L, activations are collected according to the `activation_mode`:

### Input mode (default)

1. Load the model and tokenizer from HuggingFace.
2. For each probe string (in batches):
   - Run a forward pass with `output_hidden_states=True`.
   - Extract hidden states at layer L: `(batch, seq_len, d)`.
   - Pool over the sequence dimension → one vector per probe: `h_i ∈ R^d`.
3. Stack all probe vectors: `H_L = (N_probes, d)`.
4. Optionally L2-normalize each row (`normalize_activations=True`, the default).
5. Compute the Gram matrix: `G_L = H_L @ H_L.T → (N_probes, N_probes)`.
6. Flatten the upper triangle: `N_probes*(N_probes+1)//2` values per layer.

### Generation mode

Steps 1–2 use `model.generate()` with `return_dict_in_generate=True, output_hidden_states=True`:

- At each decoding step `t`, the last-token hidden state at layer L is extracted: `h_{i,t} ∈ R^d`.
- These are accumulated across all `max_new_tokens` steps and averaged: `h_i = mean_t(h_{i,t})`.
- The resulting `(N_probes, d)` matrix is then processed identically to input mode.

This captures how the model's internal geometry evolves *while generating*, rather than while encoding the prompt.

### Both mode

Runs both input and generation passes for every probe. The per-layer Gram rows from each pass are concatenated, producing a representation with `2 × N_layers` rows:

```
rows 0 … N_layers-1       : input activation Gram rows
rows N_layers … 2*N_layers-1 : generation activation Gram rows
```

The existing distance metrics operate on the full matrix without modification.

### Final representation shape

| Mode | Shape |
|---|---|
| `"input"` or `"generation"` | `(N_layers, N_probes*(N_probes+1)//2)` |
| `"both"` | `(2*N_layers, N_probes*(N_probes+1)//2)` |

Example: 4 layers, 50 probes, `"both"` mode → `(8, 1275)`.

---

## Configuration

```python
from src import FunctionalTaxonomy

taxonomy = FunctionalTaxonomy(
    probes=probes,                        # shared input strings
    layer_indices=[-4, -3, -2, -1],       # which transformer layers to analyze
    cache=DiskCache("./cache"),           # safetensors format by default
    device="cuda",
    batch_size=8,
    torch_dtype=torch.bfloat16,           # use bfloat16 for Llama/Gemma
    hf_token=None,                        # falls back to HF_TOKEN env var
    pooling="mean",                       # how to reduce seq_len dimension
    normalize_activations=True,           # L2-normalize before computing Gram
    activation_mode="input",              # "input", "generation", or "both"
    max_new_tokens=32,                    # tokens to generate (used when mode != "input")
)
```

### `activation_mode`

Controls which phase of the model's computation is captured:

```python
# Default: forward pass activations on the input prompt only
FunctionalTaxonomy(..., activation_mode="input")

# Activations during auto-regressive generation (generates max_new_tokens tokens)
FunctionalTaxonomy(..., activation_mode="generation", max_new_tokens=50)

# Both phases stacked: (2*N_layers, features) matrix
FunctionalTaxonomy(..., activation_mode="both", max_new_tokens=50)
```

`max_new_tokens` is ignored when `activation_mode="input"`.

Changing `activation_mode` or `max_new_tokens` produces a different cache key automatically.

### `layer_indices`

Indices into the `hidden_states` tuple returned by the model, which includes the embedding layer at index 0 followed by each transformer block:

| Index | Meaning |
|---|---|
| `0` | Embedding layer output (before any transformer blocks) |
| `1` | Output of first transformer block |
| `-1` | Output of last transformer block |
| `[-4, -3, -2, -1]` | Last 4 transformer blocks |

For comparing models architecturally, use consecutive indices from the last few layers. For a quick single-layer comparison, `[-1]` is a reasonable default.

### `pooling`

Controls how the `(seq_len, d)` hidden state tensor is reduced to a single `(d,)` vector per probe:

| Option | Description |
|---|---|
| `"mean"` | Average over all token positions (recommended, robust) |
| `"last_token"` | The final token's representation; useful for causal LMs |
| `"cls"` | The `[CLS]` token; only meaningful for BERT-style masked LMs |

In generation mode, `pooling` applies to the prompt's hidden states at step 0 (which has sequence length > 1). For steps 1+ (with KV cache, sequence length = 1), the single token position is used directly.

### `normalize_activations`

When `True` (default), each row of `H_L` is divided by its L2 norm before computing the Gram matrix. This:
- Makes `G_L[i, i] = 1` for all probes (correlation matrix, not covariance).
- Removes scale differences between layers and models.
- Makes the distance metric depend only on the *direction* of activations, not their magnitude.

Set to `False` if you want to capture magnitude differences as well.

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

Computes CKA between the `(N_layers, ...)` matrices. More invariant to linear transformations than Frobenius.

```python
from src import CKADistanceMetric
metric = CKADistanceMetric(kernel="linear", unbiased=False)
```

**Important:** `CKADistanceMetric(unbiased=True)` requires the number of *rows* in the representation matrix to be at least 4. In `"both"` mode the row count is `2 * N_layers`, so 2 layers suffices. In `"input"` or `"generation"` mode, you need at least 4 layers. Use `unbiased=False` for smaller configurations.

---

## Layerwise structure

Each row of the representation matrix corresponds to one layer (or one phase-layer combination in `"both"` mode). This means:

- Models with the same architecture and the same `layer_indices` always produce matrices of the same shape.
- The distance metric operates on the full layerwise structure.
- Comparing across architectures (different numbers of layers) requires selecting layers by absolute index consistently.

In `"both"` mode, the `metadata["activation_mode"]` field records which mode was used and `metadata["n_layers"]` gives the full row count (including both phases).

---

## Probe set design

- Use diverse probes covering the input distribution of interest.
- Avoid too-similar probes — near-identical probes produce near-rank-1 Gram matrices and near-zero distances.
- 50–200 probes is a reasonable range. The Gram matrix has `N*(N+1)/2` entries per layer; for 100 probes and 4 layers this is ~20 200 float32 values per model.

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

ds = load_dataset("cais/mmlu", "all", split="test[:50]")
probes = [row["question"] for row in ds]

# Option A: input activations only (fastest)
taxonomy = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-4, -3, -2, -1],
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=8,
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
    activation_mode="input",
)

# Option B: generation activations only
taxonomy_gen = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-4, -3, -2, -1],
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=4,
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
    activation_mode="generation",
    max_new_tokens=50,
)

# Option C: both — (8, features) matrix per model
taxonomy_both = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-4, -3, -2, -1],
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=4,
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
    activation_mode="both",
    max_new_tokens=50,
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
