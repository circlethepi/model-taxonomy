# Behavioral Taxonomy

The behavioral taxonomy compares models by what they *produce* — specifically, by the semantic geometry of their generated outputs over a shared set of probe inputs. It is the most externally observable taxonomy because it relies only on what a model generates, with no access to internal weights or activations.

> **Scope boundary:** `BehavioralTaxonomy` operates exclusively on generated text. It does not collect hidden states or logits. If you want to compare models by their internal activation structure, use `FunctionalTaxonomy` instead.

## How it works

For each model in the collection:

1. Load the model and tokenizer from HuggingFace.
2. For each probe string (processed in batches):
   - Run `model.generate()` with `max_new_tokens` steps and `num_return_sequences=replicates`.
   - Decode each generated continuation.
   - Pass each generated text to an `Embedder` to get a vector `e ∈ R^d`.
3. Stack the vectors into a matrix `M ∈ R^{(N × R) × d}`, **query-major**: rows are
   `q0r0, q0r1, …, q0r(R-1), q1r0, …`. At `replicates=1` this is the familiar
   `N × d`, one row per probe.
4. Delete the model from GPU memory; clear the CUDA cache.
5. Return `ModelRepresentation(matrix=M, metadata={"generated_texts": [...]}, ...)`.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` so you can audit what the model produced without re-running extraction. They are nested per probe: `generated_texts[q][r]` is replicate `r` of probe `q`.

### Replicates and sampling

Decoding **samples** by default (`do_sample=True`). That is what makes `replicates > 1` meaningful: greedy decoding is deterministic, so R greedy replicates would be R copies of one continuation, and asking for that combination raises a `ValueError` rather than quietly storing duplicates.

Replicates exist to separate two things a single sample confounds: how far apart two models are, and how far apart two samples from *one* model are. Read them with `replicate_reduction="all"` to let a distance include the within-probe spread, or `"mean"` to average each probe's replicates back to one row and recover exactly the shape a single-sample run produced.

**Reproducibility is conditional on `batch_size`.** One RNG stream serves a whole `generate` call, so batch shape determines which tokens are drawn. A re-run at the same `batch_size` and `generation_seed` reproduces byte for byte; at a different `batch_size` it does not. `batch_size` is deliberately *not* part of the cache key — it is a machine detail, not a result — but it is recorded in `metadata` and in the `runs/` record so a mismatch is detectable after the fact. Under the previous greedy decoding this was only a last-bit effect (measured: 6/8 continuations byte-identical between batch 1 and batch 8); it is now first-order.

Note also that `replicates` multiplies the effective batch: `generate` sees `batch_size × replicates` sequences, and KV-cache growth scales with that product times `(prompt + max_new_tokens)`.

## Configuration

```python
BehavioralTaxonomy(
    queries=probes,
    embedder=embedder,
    cache=DiskCache("./cache"),   # safetensors format by default
    device="cuda",                # or "cpu"
    batch_size=8,                 # probes per generate call
    max_new_tokens=64,            # required: must be > 0
    replicates=1,                 # continuations drawn per probe
    do_sample=True,               # False = greedy; incompatible with replicates > 1
    temperature=1.0,              # 1.0 + top_p 1.0 = the model's own distribution
    top_p=1.0,                    # < 1.0 truncates the tail
    top_k=None,                   # None = no top-k truncation
    generation_seed=0,            # reproduces a run at the SAME batch_size
    torch_dtype=torch.float16,    # use bfloat16 for Llama/Gemma
    hf_token=None,                # falls back to HF_TOKEN env var
)
```

`max_new_tokens` must be greater than zero — behavioral comparison is defined by what models generate. If you pass `max_new_tokens=0`, a `ValueError` is raised at construction time with a pointer to `FunctionalTaxonomy`.

Set `batch_size` based on the model size and GPU memory. A 7B model at float16 needs ~14 GB; with a batch of 8 probes the generated token buffers add ~1–2 GB — and remember the batch `generate` actually sees is `batch_size × replicates`, so raising `replicates` without lowering `batch_size` raises memory proportionally.

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

Representations are cached to disk when a `GeneratedTextCache` is passed. An entry is addressed by `{base}/{adapter}/{recipe_hash}/n{n}_s{seed}/`, and within a draw by the filename:

```
generations/generation{max_new_tokens}_{replicates}r_{sampling_hash}.json
embeddings/generation{max_new_tokens}_{replicates}r_{sampling_hash}_{embedder_hash}.safetensors
```

Everything that changes the stored numbers is in that name, so `ls embeddings/` is a complete description of what has been computed over a draw. `sampling_hash` is an 8-hex digest of `{do_sample, temperature, top_p, top_k, generation_seed}`; it is in the name because `save()` is idempotent *on the filename*, so without it a second run at a different temperature would silently return the first run's numbers.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` inside the cache entry, so you can inspect what was generated without re-running the model — and because text lives in a plain JSON beside the tensors, auditing it needs no safetensors load.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` inside the cache entry, so you can inspect what was generated without re-running the model.

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
    queries=probes,
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
