# Behavioral Taxonomy

The behavioral taxonomy compares models by what they *produce* — specifically, by the geometric structure of their outputs over a shared set of probe inputs. It is the most generally applicable taxonomy because it requires no access to model internals beyond what the model exposes through its forward pass.

## How it works

For each model in the collection:

1. Load the model and tokenizer from HuggingFace.
2. For each probe string (processed in batches):
   - Run a forward pass with `output_hidden_states=True`.
   - Optionally generate text (`max_new_tokens > 0`).
   - Pass the output to an `Embedder` to get a vector `e ∈ R^d`.
3. Stack the `N` probe vectors into a matrix `M ∈ R^{N × d}`.
4. Delete the model from GPU memory; clear the CUDA cache.
5. Return `ModelRepresentation(matrix=M, ...)`.

This matrix is the model's *behavioral surrogate* — a compact description of how it responds to the probe distribution.

## Configuration

```python
BehavioralTaxonomy(
    probes=probes,
    embedder=embedder,
    cache=DiskCache("./cache"),   # optional, but recommended
    device="cuda",                # or "cpu"
    batch_size=8,                 # probes per forward pass
    max_new_tokens=0,             # 0 = no generation (faster)
    torch_dtype=torch.float16,    # use bfloat16 for Llama/Gemma
    hf_token=None,                # falls back to HF_TOKEN env var
)
```

Set `batch_size` based on the model size and GPU memory. A 7B model at float16 needs ~14 GB; with a batch of 8 probes of average length 20 tokens, the activations add another ~2 GB.

## Embedder strategies

The `Embedder` controls how the model output for one probe is converted into a single vector. Two strategies are available, and they can be mixed across experiments since the resulting representations are stored separately in the cache.

### Strategy 1: Hidden state embedder

Uses the model's own internal representations. No second model is needed.

```python
from src import HiddenStateEmbedder

# Default: last layer, mean pooling over sequence positions
embedder = HiddenStateEmbedder(
    strategy="hidden_states",
    layer_index=-1,          # -1 = last layer, -2 = second-to-last, etc.
    pooling="mean",          # "mean", "last_token", or "cls"
)

# Use logit vectors instead of hidden states
embedder = HiddenStateEmbedder(
    strategy="logits",
    pooling="mean",          # mean over sequence positions → (vocab_size,) vector
)
```

**When to use:** When you want to capture internal representational geometry — closer to the *functional* than the *behavioral* level, but faster to compute since no second model is needed.

**Pooling guidance:**
- `"mean"` — averages over all token positions; robust and generally recommended.
- `"last_token"` — the final token's representation; useful for causal LMs where the last token aggregates the full context.
- `"cls"` — the `[CLS]` token; only meaningful for BERT-style masked LMs.

### Strategy 2: Sentence transformer embedder

Generates text with the model, then encodes the generated text with a separate sentence-transformers model running on CPU.

```python
from src import SentenceTransformerEmbedder

embedder = SentenceTransformerEmbedder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="cpu",
    use_generated_text=True,    # False = embed the raw probe instead
    normalize_embeddings=True,
)
```

This strategy requires `max_new_tokens > 0` in `BehavioralTaxonomy`.

**When to use:** When you care about the *semantic content* of outputs rather than the model's internal representations. Two models that say the same thing in different words will be close; two models that express the same idea with different phrasing will be closer than with hidden-state embeddings.

**Sentence-transformer model:** The model is loaded once at construction time on CPU and kept alive. It does not occupy GPU memory. The default `all-MiniLM-L6-v2` is a 22M-parameter model that produces 384-dimensional embeddings; it is fast and accurate for most purposes.

## Designing probe sets

The quality of the behavioral comparison depends heavily on the probe set. Probes should:

- **Cover the relevant input distribution.** If you are comparing instruction-tuned models, include instruction-style prompts. If comparing domain specialists, use domain-specific probes.
- **Be diverse enough to prevent rank collapse.** If all probes are very similar, the representation matrices will be nearly identical for all models and distances will be near zero.
- **Be long enough to elicit meaningful responses.** Single-token probes often produce degenerate distributions.
- **Be the same for all models.** The representations are only comparable when computed on the same probe set. The `TaxonomyAnalyzer` will raise an error if the shape of two representations does not match.

A reasonable starting point is 50–200 probes drawn from a benchmark dataset (e.g., MMLU, HellaSwag) or a curated set covering the capabilities you care about.

```python
from datasets import load_dataset

# Use the first 100 questions from MMLU as probes
ds = load_dataset("cais/mmlu", "all", split="test[:100]")
probes = [row["question"] for row in ds]
```

## Caching

Representations are cached to disk by default when a `DiskCache` is passed. The cache key encodes the model ID, probe list, layer index, pooling strategy, and all other extraction parameters. Changing any one of these produces a different key and triggers fresh inference.

```python
cache = DiskCache("./cache")           # NPZ format — portable, compatible with numpy
cache = DiskCache("./cache", format="pt")  # PyTorch format — preserves bfloat16 precision
```

The cache is safe for concurrent use across SLURM jobs sharing a network filesystem — see [Compute Backends](compute_backends.md).

## Memory management

`BehavioralTaxonomy` always runs one model at a time. After `extract()` returns:

- The model is deleted from Python's object graph.
- `torch.cuda.empty_cache()` is called to release GPU memory back to the allocator.

When using `LocalBackend(n_jobs=1)`, models are processed sequentially and GPU memory usage stays bounded. Do not use `n_jobs > 1` with GPU models — multiple models cannot share a single GPU without careful memory management.

## Full example

```python
import torch
from src import (
    BehavioralTaxonomy, HiddenStateEmbedder, SentenceTransformerEmbedder,
    DiskCache, ModelCollection, TaxonomyAnalyzer,
    CKADistanceMetric, MDSGeometry, LocalBackend,
)

models = ModelCollection.from_ids([
    "meta-llama/Llama-3.2-1B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-1.5B",
])

probes = [...]  # 100 probe strings

# Option A: own hidden states (no generation needed)
taxonomy = BehavioralTaxonomy(
    probes=probes,
    embedder=HiddenStateEmbedder(layer_index=-1, pooling="mean"),
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=4,
    torch_dtype=torch.bfloat16,     # Llama uses bfloat16
    hf_token="hf_...",
)

# Option B: sentence-transformer on generated text
taxonomy_b = BehavioralTaxonomy(
    probes=probes,
    embedder=SentenceTransformerEmbedder(use_generated_text=True),
    cache=DiskCache("./cache"),
    device="cuda",
    batch_size=4,
    max_new_tokens=64,              # required for text generation
    torch_dtype=torch.bfloat16,
    hf_token="hf_...",
)

result = TaxonomyAnalyzer(
    taxonomy=taxonomy,
    metric=CKADistanceMetric(kernel="linear"),
    geometry_method=MDSGeometry(n_components=2),
    backend=LocalBackend(n_jobs=1),
).fit(list(models))
```
