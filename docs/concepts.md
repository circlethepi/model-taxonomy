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

The `cache_key` is stable: the same model + same configuration always produces the same key. Changing any extraction parameter (probe list, layer, pooling strategy) produces a different key and invalidates cached results automatically.

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
profile.add(structural_result)    # when implemented

profile.taxonomy_names()          # ["behavioral", "structural"]
profile.get("behavioral")         # TaxonomyAnalysis
profile.save("./results/full_profile")
```

---

## Caching

`DiskCache` avoids re-running inference for model-taxonomy pairs that have already been computed. It is keyed by a hash of the model ID plus all extraction parameters, so:

- Re-running with the same config hits the cache.
- Changing any parameter (probe list, layer index, pooling, etc.) misses the cache and triggers fresh inference.
- Multiple SLURM jobs writing the same key are safe: `filelock` ensures only one job writes, and atomic `os.replace()` prevents partial reads.

```python
cache = DiskCache("./cache")         # NPZ format by default
cache = DiskCache("./cache", format="pt")  # use .pt for bfloat16 precision
```

---

## Design principles

**Each step is independently swappable.** You can change the distance metric without re-running inference. You can change the geometry method without recomputing distances.

**The `Taxonomy` is self-contained.** All configuration is baked into the object at construction time so it is pickle-safe for SLURM serialization. The `ComputeBackend` calls `taxonomy.extract(model_id)` as a pure function.

**GPU memory is bounded.** `BehavioralTaxonomy.extract()` loads the model, processes all probes, then explicitly deletes the model and clears the CUDA cache before returning. Only one model occupies GPU memory at a time when using `LocalBackend(n_jobs=1)`.

**HuggingFace is the only model interface.** All models are referenced by HuggingFace Hub path and loaded via `transformers.AutoModelForCausalLM`. Authentication for gated models uses the `HF_TOKEN` environment variable or an explicit token passed to the taxonomy.
