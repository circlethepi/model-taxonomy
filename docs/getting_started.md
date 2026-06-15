# Getting Started

## Installation

The package requires Python 3.11+ and PyTorch. Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd model-taxonomy
conda env create -f environment.yml
conda activate taxonomy
pip install -e .
```

For UMAP support (optional):

```bash
pip install umap-learn
```

For SLURM cluster support:

```bash
pip install submitit
```

## Minimal example

The following runs a full functional analysis on three small public models using activation Gram matrices, CKA distances, and MDS coordinates. No GPU is required for tiny models.

```python
from src import (
    ModelCollection,
    FunctionalTaxonomy,
    CKADistanceMetric,
    MDSGeometry,
    LocalBackend,
    DiskCache,
    TaxonomyAnalyzer,
)

# 1. Define the model collection
models = ModelCollection.from_ids([
    "sshleifer/tiny-gpt2",
    "hf-internal-testing/tiny-random-GPTNeoXForCausalLM",
    "hf-internal-testing/tiny-random-OPTForCausalLM",
])

# 2. Define probe inputs
probes = [
    "The capital of France is",
    "Water boils at 100 degrees",
    "The quick brown fox",
    "To install Python, you need to",
    "She walked into the room and",
]

# 3. Configure the three pipeline steps
taxonomy = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-1],           # last transformer layer
    cache=DiskCache("./cache"),   # safetensors format by default
    device="cpu",                 # use "cuda" for GPU
)
metric = CKADistanceMetric(kernel="linear", unbiased=False)
geometry = MDSGeometry(n_components=2)

# 4. Run the pipeline
analyzer = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=metric,
    geometry_method=geometry,
    backend=LocalBackend(n_jobs=1),
)
result = analyzer.fit(list(models))

# 5. Inspect results
print(result.distance_matrix.matrix)
# [[0.    0.312 0.289]
#  [0.312 0.    0.298]
#  [0.289 0.298 0.   ]]

print(result.distance_matrix.sorted_neighbors("sshleifer/tiny-gpt2"))
# [('hf-internal-testing/tiny-random-OPTForCausalLM', 0.289),
#  ('hf-internal-testing/tiny-random-GPTNeoXForCausalLM', 0.312)]

print(result.geometry.coordinates)
# [[-0.21  0.04]
#  [ 0.18 -0.11]
#  [ 0.03  0.07]]

# 6. Save and reload (all files written as safetensors)
result.save("./results/run1")
```

## Choosing a taxonomy

| Taxonomy | What it captures | When to use |
|---|---|---|
| `FunctionalTaxonomy` | Covariance structure of internal activations | Comparing how models *process* inputs internally |
| `BehavioralTaxonomy` | Semantic content of generated text | Comparing what models *produce* |
| `StructuralTaxonomy` | LoRA adapter weight geometry | Comparing fine-tuned variants of a shared base model |

## Behavioral example (generated output)

`BehavioralTaxonomy` requires `max_new_tokens > 0` — it compares models purely on their generated text.

```python
from src import BehavioralTaxonomy, SentenceTransformerEmbedder, DiskCache

taxonomy = BehavioralTaxonomy(
    probes=probes,
    embedder=SentenceTransformerEmbedder(use_generated_text=True),
    cache=DiskCache("./cache"),
    device="cuda",
    max_new_tokens=64,
)
```

## Using a HuggingFace Hub search

Instead of listing model IDs manually, you can search the Hub:

```python
models = ModelCollection.from_hub_search(
    task="text-generation",
    library="transformers",
    limit=10,
)
```

## Using gated models

For models that require authentication (Llama, Gemma, etc.), pass your HuggingFace token:

```python
import os

taxonomy = FunctionalTaxonomy(
    probes=probes,
    layer_indices=[-4, -3, -2, -1],
    hf_token=os.environ["HF_TOKEN"],   # or pass the string directly
)
```

Alternatively, set the `HF_TOKEN` environment variable and omit the argument — all taxonomy classes read it automatically.

## Next steps

- [Core Concepts](concepts.md) — understand the data model before running larger experiments
- [Behavioral Taxonomy](guides/behavioral_taxonomy.md) — generated-output comparison and embedder strategies
- [Functional Taxonomy](guides/functional_taxonomy.md) — activation modes: input, generation, or both
- [Structural Taxonomy](guides/structural_taxonomy.md) — LoRA adapter caching and config.json
- [Compute Backends](guides/compute_backends.md) — scaling to a SLURM cluster
