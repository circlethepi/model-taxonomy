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

## The taxonomy levels

Each taxonomy captures a distinct level of abstraction:

| Taxonomy | What it extracts | Input required |
|---|---|---|
| **Structural** | LoRA adapter weight geometry | None (parameters only) |
| **Functional** | Covariance structure of internal activations | Probe strings |
| **Log-probability** | Per-token log-probabilities and entropies | Probe strings (teacher-forced; no decoding) |
| **Behavioral** | Semantic content of generated text | Probe strings + generation |
| **Dataset Embedding** | Text distribution of the fine-tuning mixture | A dataset recipe (no model) |

**Structural** captures *what was changed* during fine-tuning (the LoRA delta matrices).
**Functional** captures *how* a model processes inputs layer by layer.
**Log-probability** captures what a model *believes* — the probability it assigned to
text, on a scale that is defined for every model over every query and needs no sampling.
**Behavioral** captures *what* a model produces. **Dataset Embedding** is the one level
that touches no model at all: it describes the training mixture itself, and is therefore
the natural stand-in for ground truth.

The middle three share one `HFInferenceTaxonomy` base class and one cache key, so a
model under a given query draw sits at the same coordinates in `04_activations`,
`05_generated` and `05a_logprobs`.

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

Several cache classes cover different storage needs. All tensor data is stored in **safetensors** format — memory-mappable, pickle-free, and fast to load.

| Class | Stage | Holds |
|---|---|---|
| `SampledDatasetCache` | `01_datasets` | Recipes and the source-row indices of each draw |
| `DatasetEmbeddingCache` | `02_dataset_embeddings` | Dataset-level surrogates |
| `LoRACache` | `03_adapters` | Raw PEFT weights and structural representations |
| `ActivationCache` | `04_activations` | Pooled hidden states per layer |
| `GeneratedTextCache` | `05_generated` | Generated text and its embeddings |
| `LogProbCache` | `05a_logprobs` | Per-token log-probabilities and entropies |
| `CollectionCache` | `06_collections` | Distance matrices and geometry results |
| `DiskCache` | — | The general-purpose flat, hash-keyed fallback |

`ActivationCache`, `GeneratedTextCache` and `LogProbCache` all derive from
`DrawKeyedCache` and so share one key; `SampledDatasetCache` and
`DatasetEmbeddingCache` are keyed by recipe alone and are model-free.

### The shared cache layout

Experiments that set the same `cache_dir` share one cache tree. Its directories carry
numeric prefixes so a directory listing reads in pipeline order:

```
results/shared_cache/
    01_datasets/                 {recipe_hash}/recipe.json + names.json + n{n}_s{seed}.json
    02_dataset_embeddings/       {recipe_hash}/n{n}_s{seed}/{embedder_hash}/surrogates/{hash}/
    03_adapters/                 raw PEFT weights + extracted structural representations
    03A_adapter_alignments/      pairwise Procrustes alignments
    04_activations/              {base}/{adapter}/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
    05_generated/                {base}/{adapter}/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
    05a_logprobs/                {base}/{adapter}/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
    06_collections/              distance matrices, geometries, index.json
```

The three inference stages share that key exactly, from
`src/cache/_draw_keyed.py::DrawKeyedCache`, so one model under one query draw
sits at the same coordinates in all of them and the trees can be read side by
side. They differ only in the artifact filename: `{mode}_{pooling}_layer{NNN}`
for activations, `{mode}_{replicates}r_{sampling_hash}_{embedder_hash}` for
generations, and `input` or the *same* generation variant token for log-probs —
`LogProbCache` reuses `GeneratedTextCache.variant_token` as the same function
object, so a log-prob file and the generation it describes join by name with no
lookup. The extra components on the generation side are what sampling cost: a
temperature and a replicate count both change the text, and neither appears
anywhere else in the path, so both are in the name or a second run silently
reuses the first one's entry.

The optional `_f{fmt}` suffix is how a **prompt format** enters the path — the
first eight hex characters of a hash over the `prompt_format:` block. Same
adapter, same recipe, same `(n, seed)`, but prompts rendered through a different
chat template is a genuinely different computation, and every save in these
stages is idempotent on filename, so without it the second run silently no-ops.
It is omitted by default, which keeps every pre-existing path byte-identical,
and it is deliberately absent from `01_datasets` and `02_dataset_embeddings`:
those are keyed by recipe alone and are genuinely model-free, and a chat
template is a property of a model, not of a draw. It is equally deliberately not
folded into `recipe_hash`, which would change the identity of every cached draw
at once. See [Model Profiles and Prompt Formats](guides/model_profiles.md).

Directories sharing a number sit at the same stage. A **letter suffix means analysis
*of* the objects at that stage** — `03A_adapter_alignments` holds things computed from
what is in `03_adapters`, not a stage of its own.

Three things about this layout are worth stating rather than leaving to be discovered:

- **Structural representations live inside `03_adapters/{adapter}/{config_hash}/`**,
  beside the weights they were derived from, while functional and behavioral
  representations get their own stage directories. That asymmetry is deliberate — a
  LoRA representation is a transformation of that adapter's own weights and nothing
  else — but it is not obvious from the numbering. `StructuralTaxonomy` therefore has
  exactly one caching protocol, `LoRACache`; without one it recomputes every call.
- **`01_datasets/{recipe_hash}/recipe.json` is the authoritative home for recipes.**
  `DatasetEmbeddingCache` also writes a copy under `02_` so the embedding cache stays
  self-describing on its own, but resolution goes through `01_`. Before this existed
  the embedding cache was the only hash-indexed source, so a recipe that was sampled
  but never embedded could not be resolved at all.
- **`recipe_hash` is content-addressed: one mixture, one directory.** It is a SHA-256
  of `{recipe_type, datasets}` — the *name is not hashed*. n and seed live in the draw
  filename, not the identity, so a mixture swept over 19 sizes and 10 seeds is one
  recipe with ~190 draws beside it rather than 190 near-duplicate recipes. Two
  consequences worth knowing: `names.json` accumulates every config-block name that
  resolved to the hash, because several legitimately do; and something had to keep
  separating seeds once the recipe hash stopped doing it, or a seed sweep would
  silently collapse onto one embedding. That was `DatasetEmbeddingCache.embedder_hash`
  until item 15 moved it into the path, where `01`, `04` and `05` had always kept it.
- **How a row becomes text is part of the recipe, therefore part of the hash.** An
  entry names either one column (`text_field`) or several composed with a separator
  (`text_fields` + `text_separator`), and `text_fields` wins wherever both appear.
  The same rows projected two different ways are two different training sets, so
  they must not share a directory — composing gives a new `recipe_hash` and a new
  draw. The composition keys are omitted from `to_dict()` when unset rather than
  written as nulls, which is what let this be added without moving the six existing
  hashes; the cost is that a composed `recipe.json` still carries a stale, unused
  `text_field`. See [Dataset recipes](api_reference.md#dataset-recipes).
- **Every stage spells a draw the same way, and `src/cache/_draw.py` owns the
  spelling.** `n{n}_s{seed:02d}`, zero-padded. It has not always been so: `01` wrote
  an unpadded seed while `04`/`05` padded theirs, and `02` wrote no draw at all — so
  the same coordinate had three spellings and nothing compared them. Writing is
  narrow and reading is wide: `draw_name` always pads, `parse_draw_name` accepts
  both, so pre-migration names still read.
- **A `02` surrogate is authored, not derived — unlike `04`/`05`.** The inference
  stages store a base artifact and compute views from it at read time, so a new view
  costs only CPU. `02` stores no base: `representation` is chosen before embedding,
  and keeping the full `(N, 768)` would cost 6.1 GB plus a GPU re-embed of ~2M texts.
  The directory shape matches; the guarantee does not. Adding a representation here
  means re-embedding, so do not write code that reconstructs one surrogate from a
  sibling.
- **Draws store source indices, not rows.** `n{n}_s{seed}.json` records which rows of
  which upstream split were drawn, in order, with the Hub revision pinned and a
  `rows_sha256` over the result — not the row text, which is already sitting in the
  HuggingFace cache. That is ~85× smaller (2.07 GiB → 40 MiB across 564 draws) and
  rehydrates faster, at the cost of a dependency on the upstream dataset that
  `scripts/verify_sampled_cache.py` exists to audit.
- **The numbering applies to the shared cache only.** `{output_dir}/adapters/` — the
  fallback when no `cache_dir` is configured — stays unnumbered, because an experiment
  output directory has no pipeline ordering to describe. Legacy per-experiment cache
  trees at `results/<experiment>/cache/` still use the old flat names and are **not**
  readable by current code; they are superseded work, left in place deliberately
  rather than migrated.

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
# ./cache/03_adapters/meta-llama--Llama-3.1-8B/some-org--my-adapter/
#     adapter_model.safetensors       ← raw PEFT weights (untouched)
#     adapter_config.json
#     {config_hash}/                  ← one dir per extraction configuration
#         config.json                 ← training details + dataset_recipe stub
#         representation.safetensors  ← extracted representation matrix
```

The `{config_hash}` level lets several extraction configurations (different
layers, projections, or `use_lora_product`) coexist for one adapter. Accordingly
`exists()`, `load()`, `load_config()` and `save()` all take an
`extraction_config: dict` identifying which one to use.

Pass it to `StructuralTaxonomy` instead of (or in addition to) `DiskCache`:

```python
taxonomy = StructuralTaxonomy(lora_cache=LoRACache("./cache"), ...)
```

### `CollectionCache` — distance matrices and geometry results

Stores the outputs of a full pipeline run — distance matrix plus any geometry embeddings — so they can be reloaded without re-running the model extraction step.

```python
from src.cache import CollectionCache

cc = CollectionCache("./cache")

# The handle names what the matrix was built from, not just which models it covers.
handle = cc.handle(
    distance_matrix.taxonomy,
    cc.collection_key(model_entries),         # models + their artifact paths
    distance_matrix.metric,
    cc.surrogate_key([e["surrogate_hash"] for e in model_entries]),
)
cc.save_distance_matrix(distance_matrix, handle, model_entries=model_entries)
cc.save_geometry(handle, geometry_result)

dm = cc.load_distance_matrix(handle)
mds = cc.load_geometry(handle, "mds", 2)
info = cc.load_info(handle)     # collection_info.json as dict
cfg = cc.load_config(handle)    # the leaf's surrogate spec and per-model hashes
```

In practice you rarely assemble a handle by hand — `build_taxonomy_artifacts`
does it from the representations it resolved, which is the point: the key is
derived from the tensors actually read, so a collection cannot be returned for a
selector it was not built with.

`collection_info.json`, at the `{collection_key}` level, records the models and
LoRA adapters and their artifact paths, shared by every metric and view over
them. Each leaf's `config.json` records the surrogate spec and the per-model
surrogate hashes — enough to trace the collection back to its inputs.

---

## Analysing the results

`src/analysis` sits on top of `DistanceMatrix` and `GeometryResult` and compares
them at three levels.

**Distance matrices** (`src.analysis.matrices`). Correlate two taxonomies'
off-diagonal vectors — for five models, two 10-element vectors, one entry per
model pair — to ask whether they rank model-pair similarity the same way.
`mantel_test` and `dcor_test` both build a null by permuting model labels jointly
across rows and columns, which destroys the correspondence between the two
matrices while leaving each one's internal structure untouched.

They differ in what they compute on it, and the difference decides which to
believe. Mantel correlates the `n(n-1)/2` off-diagonal entries, which are derived
from `n` points and therefore dependent — its p-value is not calibrated and is
kept only as a descriptive statistic. `dcor_test` works on the doubly-centred
matrices instead, so the permutation null is the whole of the inference, and it
reads the matrices directly rather than an embedding of them (which is what
separates it from `protest` at the configuration level).

Neither replaces the other. dCor measures *dependence*, so it is blind to
direction: a taxonomy recovering the mixing order exactly backwards scores 1.0.
The signed correlation is what tells the two apart, which is why both are
reported.

**Fleet-level surrogates** (`src.analysis.surrogates`). A `DistanceMetric` sees
exactly two models, so it can never subtract a fleet mean or divide by a fleet
covariance — those are collection-level operations. `center_representations` and
`whiten_representations` apply them between resolution and distancing, tagging
what they did in `metadata["surrogate_transform"]`. Every level here carries a
large component identical across all models by construction (the same questions,
the same answer register, the same base model), and removing it is what leaves
the between-model variation a taxonomy is trying to measure. See
[Cross-Level Comparison](guides/cross_level_comparison.md).

**Point configurations** (`src.analysis.configurations`). An embedding fixes
coordinates only up to rotation, reflection, translation and scale, and picks
among those arbitrarily. Procrustes superposition quotients that out, so
`procrustes_compare` depends on shape alone. `per_point_residuals` then says
*which* models the two taxonomies disagree about, and `point_dispersion` says
which models sit stably across seeds.

**Anchor simplices** (`src.analysis.simplex`). Choose `k` models as anchors and
write every model in barycentric coordinates of the simplex they span — with two
anchors, a position along the segment between them, plus a residual for how far
off that line the model actually sits.

Barycentric coordinates are the natural currency for comparing geometries: for a
point in the anchors' affine hull they are invariant under *any* invertible
affine map, and for points off the hull the least-squares projection is invariant
under similarity transforms — which is exactly the ambiguity an embedding leaves.
Two geometries from different taxonomies are therefore directly comparable in
barycentric form, with no Procrustes step in between.

This is also where the pipeline can be checked against something it never saw.
Every other measure here compares one derived quantity to another; with adapters
fine-tuned on known topic mixtures, `anchor_weight_vs_truth` compares the
recovered mixing proportion against the real one.

`src.analysis.ground_truth` scores a whole taxonomy against that truth two ways.
`dcor_vs_truth` scores the **distances** and never embeds, so it is untouched by
MDS distortion; `disparity_vs_truth` scores the **configuration** an embedding
actually draws, and so inherits whatever distortion the projection introduced.
The two come apart — a taxonomy can reproduce the distance profile while
arranging the points in something that is not the simplex — and they run in
opposite directions: higher dCor is better, lower disparity is better.

Distance matrices computed directly from LoRA factors (`src.notebook.structure`)
enter the same layer via `src.analysis.lora_distance_matrix`, which returns an
ordinary `DistanceMatrix` — so notebook work and pipeline runs are analysed and
plotted with one set of tools.

### Matching identifiers across levels

The model-level taxonomies key their rows by **adapter path**
(`results/yahoo_topics/adapters/meta-llama--Llama-3.2-3B/yahoo_topic0_only_r16`);
`DatasetEmbeddingTaxonomy` keys its rows by **recipe ID** (`yahoo_topic0_only`),
because its objects are datasets, not models. Those are the same experimental
thing seen from two sides, but as strings they never match, so comparing the two
levels directly finds nothing in common.

`src.analysis.recipe_id_for` maps an adapter onto the recipe it was trained on —
authoritatively, from the `dataset_name` that `finetune_lora.py` records in
`experiment_meta.json` beside every adapter. Pass it as `key=` to any comparison
function:

```python
from src.analysis import correlation_table, recipe_id_for

labels, table = correlation_table(profile, key=recipe_id_for)
```

Dataset-level entries with no corresponding adapter — a held-out probe set, say
— simply fall out of the intersection. Nothing on disk is rewritten; use
`relabel()` if you want a copy carrying the new identifiers. Because the mapping
is many-to-one (rank and seed variants of one dataset collapse together), a
collection that sweeps those parameters must not be relabelled this way, and
`relabel()` raises rather than let distinct models collide.

---

## Design principles

**Each step is independently swappable.** You can change the distance metric without re-running inference. You can change the geometry method without recomputing distances.

**The `Taxonomy` is self-contained.** All configuration is baked into the object at construction time so it is pickle-safe for SLURM serialization. The `ComputeBackend` calls `taxonomy.extract(model_id)` as a pure function.

**GPU memory is bounded.** Each taxonomy loads the model, processes all probes, then explicitly deletes the model and clears the CUDA cache before returning. Only one model occupies GPU memory at a time when using `LocalBackend(n_jobs=1)`.

**Behavioral vs functional is a strict boundary.** `BehavioralTaxonomy` operates only on *generated text* — it never reads hidden states or logits. For activation-based comparison, use `FunctionalTaxonomy`. This boundary ensures the cache keys and representations are semantically meaningful.

**HuggingFace is the only model interface.** All models are referenced by HuggingFace Hub path and loaded via `transformers.AutoModelForCausalLM`. Authentication for gated models uses the `HF_TOKEN` environment variable or an explicit token passed to the taxonomy.
