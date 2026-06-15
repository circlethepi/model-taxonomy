# Behavioral Taxonomy

The behavioral taxonomy compares models by what they *produce* — specifically, by the semantic geometry of their generated outputs over a shared set of probe inputs. It is the most externally observable taxonomy because it relies only on what a model generates, with no access to internal weights or activations.

> **Scope boundary:** `BehavioralTaxonomy` operates exclusively on generated text. It does not collect hidden states or logits. If you want to compare models by their internal activation structure, use `FunctionalTaxonomy` instead.

## How it works

For each model in the collection:

1. Load the model and tokenizer from HuggingFace.
2. For each probe string (processed in batches):
   - Run `model.generate()` with `max_new_tokens` steps.
   - Decode the generated continuation.
   - Pass the generated text to an `Embedder` to get a vector `e ∈ R^d`.
3. Stack the `N` probe vectors into a matrix `M ∈ R^{N × d}`.
4. Delete the model from GPU memory; clear the CUDA cache.
5. Return `ModelRepresentation(matrix=M, metadata={"generated_texts": [...]}, ...)`.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` so you can audit what the model produced without re-running extraction.

## Configuration

```python
BehavioralTaxonomy(
    probes=probes,
    embedder=embedder,
    cache=DiskCache("./cache"),   # safetensors format by default
    device="cuda",                # or "cpu"
    batch_size=8,                 # probes per generate call
    max_new_tokens=64,            # required: must be > 0
    torch_dtype=torch.float16,    # use bfloat16 for Llama/Gemma
    hf_token=None,                # falls back to HF_TOKEN env var
)
```

`max_new_tokens` must be greater than zero — behavioral comparison is defined by what models generate. If you pass `max_new_tokens=0`, a `ValueError` is raised at construction time with a pointer to `FunctionalTaxonomy`.

Set `batch_size` based on the model size and GPU memory. A 7B model at float16 needs ~14 GB; with a batch of 8 probes the generated token buffers add ~1–2 GB.

## Embedder strategy

The `Embedder` controls how the generated text for one probe is converted into a single vector.

### Sentence transformer embedder

Encodes the generated text with a separate sentence-transformers model running on CPU.

```python
from src import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    use_generated_text=True,      # embeds the generated continuation
    normalize_embeddings=True,
)
```

**When to use:** When you care about the *semantic content* of outputs — two models that express the same idea in different words will be similar, while models that produce factually different answers will be far apart.

**Sentence-transformer model:** Loaded once at construction time on CPU and kept alive across all models. It does not occupy GPU memory. The default `all-MiniLM-L6-v2` is a 22M-parameter model producing 384-dimensional embeddings — fast and accurate for most purposes.

**Setting `use_generated_text=False`** embeds the raw probe string instead of the generated text. This makes behavioral distances probe-distribution-only and effectively constant across models; it is rarely useful but available as a baseline.

---

> **Note on `HiddenStateEmbedder`:** Passing a `HiddenStateEmbedder` to `BehavioralTaxonomy` will raise a `ValueError` when `embed()` is called, because `BehavioralTaxonomy` does not collect hidden states. Use `FunctionalTaxonomy` if you want to compare activation-based representations — it provides the same layerwise control with a cleaner interface.

---

## Designing probe sets

The quality of the behavioral comparison depends heavily on the probe set. Probes should:

- **Cover the relevant input distribution.** If you are comparing instruction-tuned models, include instruction-style prompts. If comparing domain specialists, use domain-specific probes.
- **Be diverse enough to prevent rank collapse.** If all probes are very similar, the representation matrices will be nearly identical for all models and distances will be near zero.
- **Be long enough to elicit meaningful responses.** Single-token probes often produce degenerate generations.
- **Be the same for all models.** The representations are only comparable when computed on the same probe set. The `TaxonomyAnalyzer` will raise an error if the shape of two representations does not match.

A reasonable starting point is 50–200 probes drawn from a benchmark dataset (e.g., MMLU, HellaSwag) or a curated set covering the capabilities you care about.

```python
from datasets import load_dataset

# Use the first 100 questions from MMLU as probes
ds = load_dataset("cais/mmlu", "all", split="test[:100]")
probes = [row["question"] for row in ds]
```

## Caching

Representations are cached to disk when a `DiskCache` is passed. The cache key encodes the model ID, probe list, embedder config, `max_new_tokens`, and all other extraction parameters. Changing any one of these produces a different key and triggers fresh inference.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` inside the cache entry, so you can inspect what was generated without re-running the model.

```python
cache = DiskCache("./cache")              # safetensors (default, fast)
cache = DiskCache("./cache", format="npz")  # NumPy zip (backward compat)
```

The cache is safe for concurrent use across SLURM jobs sharing a network filesystem.

## Memory management

`BehavioralTaxonomy` always runs one model at a time. After `extract()` returns:

- The model is deleted from Python's object graph.
- `torch.cuda.empty_cache()` is called to release GPU memory back to the allocator.

When using `LocalBackend(n_jobs=1)`, models are processed sequentially and GPU memory usage stays bounded. Do not use `n_jobs > 1` with GPU models.

## Full example

```python
import torch
from datasets import load_dataset
from src import (
    BehavioralTaxonomy, SentenceTransformerEmbedder,
    DiskCache, ModelCollection, TaxonomyAnalyzer,
    CKADistanceMetric, MDSGeometry, LocalBackend,
)

models = ModelCollection.from_ids([
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B",
])

ds = load_dataset("cais/mmlu", "all", split="test[:100]")
probes = [row["question"] for row in ds]

taxonomy = BehavioralTaxonomy(
    probes=probes,
    embedder=SentenceTransformerEmbedder(use_generated_text=True),
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=4,
    max_new_tokens=64,
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=CKADistanceMetric(kernel="linear", unbiased=False),
    geometry_method=MDSGeometry(n_components=2),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))

# Inspect nearest neighbors by behavioral output similarity
print(result.distance_matrix.sorted_neighbors("meta-llama/Llama-3.2-1B"))

# Audit what was actually generated
rep = result.representations[0]
for probe, text in zip(probes[:3], rep.metadata["generated_texts"][:3]):
    print(f"  Q: {probe[:60]}")
    print(f"  A: {text[:80]}")
    print()
```
