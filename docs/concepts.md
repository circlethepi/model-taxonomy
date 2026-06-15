# Core Concepts

## The pipeline

Every analysis is a composition of three independently configurable steps:

### Step 1 — Surrogate extraction (`Taxonomy`)

A `Taxonomy` defines *what* information to extract from a model. For a given model ID, it loads the model, runs inference, and returns a `ModelRepresentation` — a matrix `M ∈ R^{N × d}` where `N` is the number of probe inputs and `d` is the embedding dimension.

```
model_id  ──────────►  Taxonomy.extract()  ──────────►  ModelRepresentation
                         (loads model,                   matrix: (N, d)
                          runs inference,
                          extracts vectors,
                          unloads model)
```

The model is fully unloaded (weights deleted, GPU cache cleared) before the call returns. This keeps memory usage bounded when processing a collection of large models sequentially.

### Step 2 — Pairwise distances (`DistanceMetric`)

A `DistanceMetric` takes two `ModelRepresentation` objects and returns a non-negative scalar. Applied across all pairs in a collection, it produces a symmetric `DistanceMatrix`.

```
[rep_a, rep_b, rep_c, ...]  ──►  DistanceMetric.compute(a, b)  ──►  DistanceMatrix
                                  (for all pairs)                     matrix: (N, N)
```

Distance is always 0 for a model compared with itself and non-negative otherwise. The two representations must have been created with the same number of probes.

### Step 3 — Coordinate embedding (`GeometryMethod`)

A `GeometryMethod` takes a `DistanceMatrix` and embeds the models into a low-dimensional space. The result is a `GeometryResult` containing an `(N, k)` coordinate array where `k` is the number of components (typically 2 or 3 for visualization).

```
DistanceMatrix  ──►  GeometryMethod.fit()  ──►  GeometryResult
                      (MDS, PCA, UMAP)           coordinates: (N, k)
```

The geometry step is optional — you can analyze the distance matrix directly without embedding.

---

## The three taxonomy levels

Each taxonomy captures a distinct level of abstraction:

| Taxonomy | What it extracts | Input required |
|---|---|---|
| **Structural** | LoRA adapter weight geometry | None (parameters only) |
| **Functional** | Covariance structure of internal activations | Probe strings |
| **Behavioral** | Semantic content of generated text | Probe strings + generation |

**Structural** captures *what was changed* during fine-tuning (the LoRA delta matrices). **Functional** captures *how* a model processes inputs layer by layer. **Behavioral** captures *what* a model produces.

---

## Data containers

### `ModelRepresentation`

The raw output of a `Taxonomy`. Stores the matrix and metadata, and carries a `cache_key` derived from the model ID and all extraction parameters.

```python
@dataclass
class ModelRepresentation:
    model_id:  str           # HuggingFace model ID or local path
    taxonomy:  str           # name of the taxonomy that created it
    matrix:    np.ndarray    # float32, shape (N_probes, embedding_dim)
    metadata:  dict          # arbitrary extraction metadata
    cache_key: str           # SHA-256 of (model_id, config_hash)
```

The `cache_key` is stable: the same model + same configuration always produces the same key. Changing any extraction parameter (probe list, layer, pooling strategy, activation mode) produces a different key and invalidates cached results automatically.

### `DistanceMatrix`

A symmetric `(N, N)` NumPy array paired with the list of model IDs that define its rows and columns.

```python
# Index by model ID pair
d = dm[("meta-llama/Llama-3.2-1B", "Qwen/Qwen2.5-1.5B")]

# Ranked list of neighbors
neighbors = dm.sorted_neighbors("meta-llama/Llama-3.2-1B")
# [("meta-llama/Llama-3.2-1B-Instruct", 0.023), ("Qwen/Qwen2.5-1.5B", 0.387), ...]
```

### `GeometryResult`

The coordinate embedding plus bookkeeping (method name, taxonomy, stress if applicable).

```python
# k nearest neighbors in coordinate space
neighbors = geo.nearest_neighbors("meta-llama/Llama-3.2-1B", k=3)

# NetworkX graph: nodes = models, edge weights = distances
g = geo.to_networkx(distance_matrix=dm)  # weights from dm
g = geo.to_networkx()                    # weights from Euclidean coord distance
```

### `TaxonomyAnalysis`

Bundles all three outputs for a single taxonomy run.

```python
result.taxonomy_name       # "behavioral"
result.model_ids           # list of model IDs
result.representations     # list of ModelRepresentation
result.distance_matrix     # DistanceMatrix
result.geometry            # GeometryResult (or None if no geometry_method was set)
result.save("./results/run1")
```

### `ModelTaxonomyProfile`

Groups multiple `TaxonomyAnalysis` objects for the same model collection, one per taxonomy level.

```python
profile = ModelTaxonomyProfile(model_ids=model_ids)
profile.add(behavioral_result)
profile.add(structural_result)

profile.taxonomy_names()          # ["behavioral", "structural"]
profile.get("behavioral")         # TaxonomyAnalysis
profile.save("./results/full_profile")
```

---

## Caching

Three cache classes cover different storage needs. All tensor data is stored in **safetensors** format — memory-mappable, pickle-free, and fast to load.

### `DiskCache` — flat hash-keyed cache

The general-purpose cache for `ModelRepresentation` objects. Keyed by a SHA-256 hash of the model ID plus all extraction parameters.

```python
cache = DiskCache("./cache")                      # safetensors format (default)
cache = DiskCache("./cache", format="npz")        # NumPy zip (backward compat)
cache = DiskCache("./cache", format="pt")         # PyTorch (preserves bfloat16)
```

Re-running with the same config hits the cache. Changing any parameter (probe list, layer index, pooling, `activation_mode`, etc.) misses the cache and triggers fresh extraction.

Files are stored at `cache_dir/{key[:2]}/{key}.safetensors`. Writes are atomic (`os.replace`) and protected by a per-key `filelock`, making concurrent SLURM writes safe.

### `LoRACache` — hierarchical LoRA adapter cache

Organises structural representations under `base_model → adapter`, alongside a human-readable `config.json` with fine-tuning details.

```python
from src.cache import LoRACache

lora_cache = LoRACache("./cache")

# Directory structure:
# ./cache/loras/meta-llama--Llama-3.1-8B/some-org--my-adapter/
#     config.json                 ← training details + dataset_recipe stub
#     representation.safetensors  ← extracted representation matrix
```

Pass it to `StructuralTaxonomy` instead of (or in addition to) `DiskCache`:

```python
taxonomy = StructuralTaxonomy(lora_cache=LoRACache("./cache"), ...)
```

### `CollectionCache` — distance matrices and geometry results

Stores the outputs of a full pipeline run — distance matrix plus any geometry embeddings — so they can be reloaded without re-running the model extraction step.

```python
from src.cache import CollectionCache

cc = CollectionCache("./cache")
chash = cc.save_distance_matrix(distance_matrix, model_entries)
cc.save_geometry(chash, geometry_result)

dm = cc.load_distance_matrix(chash)
pca = cc.load_geometry(chash, "pca")
info = cc.load_info(chash)   # collection_info.json as dict
```

`collection_info.json` records the models and LoRA adapters in the collection, the metric and taxonomy used, and the list of geometry methods computed — enough to reconstruct the collection from scratch if needed.

---

## Design principles

**Each step is independently swappable.** You can change the distance metric without re-running inference. You can change the geometry method without recomputing distances.

**The `Taxonomy` is self-contained.** All configuration is baked into the object at construction time so it is pickle-safe for SLURM serialization. The `ComputeBackend` calls `taxonomy.extract(model_id)` as a pure function.

**GPU memory is bounded.** Each taxonomy loads the model, processes all probes, then explicitly deletes the model and clears the CUDA cache before returning. Only one model occupies GPU memory at a time when using `LocalBackend(n_jobs=1)`.

**Behavioral vs functional is a strict boundary.** `BehavioralTaxonomy` operates only on *generated text* — it never reads hidden states or logits. For activation-based comparison, use `FunctionalTaxonomy`. This boundary ensures the cache keys and representations are semantically meaningful.

**HuggingFace is the only model interface.** All models are referenced by HuggingFace Hub path and loaded via `transformers.AutoModelForCausalLM`. Authentication for gated models uses the `HF_TOKEN` environment variable or an explicit token passed to the taxonomy.
