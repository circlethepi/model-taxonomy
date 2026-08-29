# API Reference

All public classes are importable directly from `src`:

```python
from src import BehavioralTaxonomy, CKADistanceMetric, MDSGeometry, ...
from src.cache import DiskCache, LoRACache, CollectionCache
from src.analysis import barycentric, mantel_test, procrustes_compare, ...
```

---

## Core types

### `ModelID`

```python
ModelID = str
```

A HuggingFace Hub path (`"meta-llama/Llama-3.2-1B"`) or a local directory path. Used everywhere a model is identified.

---

### `ModelRepresentation`

```python
@dataclass
class ModelRepresentation:
    model_id:  ModelID
    taxonomy:  str
    matrix:    np.ndarray    # float32, shape (N_probes, embedding_dim)
    metadata:  dict
    cache_key: str

    # Properties
    n_queries: int
    embedding_dim: int

    # Factory
    @classmethod
    def create(
        cls,
        model_id: ModelID,
        taxonomy: str,
        matrix: np.ndarray,
        config: dict,
        metadata: dict | None = None,
    ) -> ModelRepresentation
```

Use `ModelRepresentation.create()` rather than the raw constructor — it computes `cache_key` automatically from `config`.

**`metadata` keys by taxonomy:**

| Taxonomy | Keys |
|---|---|
| `structural` | `n_layers`, `layer_labels`, `lora_only` |
| `functional` | `n_queries`, `n_layers`, `layer_indices`, `activation_mode`, `representation` |
| `behavioral` | `n_queries`, `generated_texts` |

---

### `DistanceMatrix`

```python
@dataclass
class DistanceMatrix:
    matrix:    np.ndarray    # float64, shape (N, N), symmetric, zero diagonal
    model_ids: list[ModelID]
    metric:    str
    taxonomy:  str

    def __getitem__(self, pair: tuple[ModelID, ModelID]) -> float
    def sorted_neighbors(self, model_id: ModelID) -> list[tuple[ModelID, float]]
    def reindex(self, model_ids: Sequence[ModelID]) -> DistanceMatrix
    def save(self, path: Path) -> None      # writes distance_matrix.safetensors
    @classmethod
    def load(cls, path: Path) -> DistanceMatrix
```

`sorted_neighbors` returns all other models sorted by ascending distance.

`reindex` returns the matrix with its rows and columns in the given order, and is the
guard on **every read from the collection cache**: `collection_key` sorts the model
entries before hashing, so row order is not part of the handle and a cache hit can come
back in whoever-wrote-it-first's order. A subset is allowed — that is how a superset
collection on disk becomes a legitimate hit — while an unknown or repeated id raises.
Permuting a symmetric distance matrix is exact; a *geometry* is not, so coordinates are
refitted rather than permuted when the ids do not match.

---

### `GeometryResult`

```python
@dataclass
class GeometryResult:
    coordinates:  np.ndarray      # float32, shape (N, n_components)
    model_ids:    list[ModelID]
    method:       str
    taxonomy:     str
    n_components: int
    stress:       float | None    # MDS stress; None for PCA/UMAP
    metadata:     dict

    def nearest_neighbors(self, model_id: ModelID, k: int = 3) -> list[ModelID]
    def to_networkx(self, distance_matrix: DistanceMatrix | None = None) -> nx.Graph
    def save(self, path: Path) -> None      # writes geometry.safetensors
    @classmethod
    def load(cls, path: Path) -> GeometryResult
```

`to_networkx` uses `distance_matrix` edge weights when provided, otherwise uses Euclidean coordinate distances.

---

### `TaxonomyAnalysis`

```python
@dataclass
class TaxonomyAnalysis:
    taxonomy_name:   str
    model_ids:       list[ModelID]
    representations: list[ModelRepresentation]
    distance_matrix: DistanceMatrix
    geometry:        GeometryResult | None

    def save(self, path: Path) -> None      # writes safetensors for all tensors
    @classmethod
    def load(cls, path: Path) -> TaxonomyAnalysis
```

---

### `ModelTaxonomyProfile`

```python
@dataclass
class ModelTaxonomyProfile:
    model_ids: list[ModelID]
    analyses:  dict[str, TaxonomyAnalysis]

    def add(self, analysis: TaxonomyAnalysis) -> None
    def get(self, taxonomy_name: str) -> TaxonomyAnalysis
    def taxonomy_names(self) -> list[str]
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> ModelTaxonomyProfile
```

---

### `TaxonomyAnalyzer`

```python
class TaxonomyAnalyzer:
    def __init__(
        self,
        taxonomy: Taxonomy,
        metric: DistanceMetric,
        backend: ComputeBackend,
        geometry_method: GeometryMethod | None = None,
    )

    def fit(self, model_ids: Sequence[ModelID]) -> TaxonomyAnalysis
```

Runs the complete three-step pipeline. `geometry_method=None` skips the coordinate embedding step.

---

## Models

### `ModelCollection`

```python
class ModelCollection:
    @classmethod
    def from_ids(cls, model_ids: list[str]) -> ModelCollection

    @classmethod
    def from_hub_search(
        cls,
        search: str | None = None,
        author: str | None = None,
        task: str | None = None,
        library: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> ModelCollection

    def metadata(self, model_id: ModelID) -> ModelInfo    # huggingface_hub.ModelInfo
    def to_list(self) -> list[ModelID]
    def __iter__(self) -> Iterator[ModelID]
    def __len__(self) -> int
```

---

## Taxonomies

All taxonomies share this abstract interface:

```python
class Taxonomy(ABC):
    @abstractmethod
    def extract(self, model_id: ModelID) -> ModelRepresentation: ...
    @property
    @abstractmethod
    def taxonomy_name(self) -> str: ...
    @abstractmethod
    def config_dict(self) -> dict[str, Any]: ...
```

### `BehavioralTaxonomy`

```python
class BehavioralTaxonomy(HFInferenceTaxonomy, Taxonomy):
    def __init__(
        self,
        queries: Sequence[str],
        embedder: Embedder,
        query_key: dict | None = None,
        cache: GeneratedTextCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 64,        # must be > 0; raises ValueError otherwise
        replicates: int = 1,             # continuations drawn per probe
        do_sample: bool = True,          # False = greedy; rejects replicates > 1
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        generation_seed: int = 0,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,     # falls back to HF_TOKEN env var
        source_indices: list | None = None,
        collect_logprobs: bool = False,  # also write generation-mode log-probs
        logprob_cache: LogProbCache | None = None,
    )
    taxonomy_name = "behavioral"
```

Compares models by the semantic content of their generated text. `max_new_tokens` must be `> 0` — behavioral comparison is defined by what models produce. Use `FunctionalTaxonomy` for activation-based comparison.

Generated texts are stored in `ModelRepresentation.metadata["generated_texts"]` for auditing.

**`HiddenStateEmbedder` is not compatible with `BehavioralTaxonomy`.** `BehavioralTaxonomy` does not collect hidden states; passing `HiddenStateEmbedder` will raise a `ValueError` when `embed()` is called. Use `SentenceTransformerEmbedder` instead.

The matrix is `(N × R, d)` and **query-major**: rows are `q0r0, q0r1, …, q1r0, …`. At
`replicates=1` this is the familiar `N × d`. `replicates > 1` with `do_sample=False`
raises rather than storing R copies of one greedy continuation.

`collect_logprobs=True` (with a `logprob_cache`) additionally records, for the tokens the
model actually drew, both the *processed* distribution they were drawn from and the
*unprocessed* model output — only the latter is comparable across a temperature sweep.
See [Log-Probability Taxonomy](guides/logprob_taxonomy.md).

---

### `FunctionalTaxonomy`

```python
class FunctionalTaxonomy(HFInferenceTaxonomy, Taxonomy):
    def __init__(
        self,
        queries: Sequence[str],
        layer_indices: list[int] | None = None,
        query_key: dict | None = None,
        cache: ActivationCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
        normalize_activations: str | bool = "layer",
        activation_mode: Literal["input", "generation", "both"] = "input",
        max_new_tokens: int = 32,
        view: Literal["concat", "gram"] = "concat",
        source_indices: list | None = None,
    )
    taxonomy_name = "functional"
```

| Parameter | Description |
|---|---|
| `query_key` | **Required.** The `{recipe_hash, n_samples, seed}` triple identifying the draw in `01_datasets`. Keys the cache; the query strings are never hashed. |
| `layer_indices` | Indices into `hidden_states`; `0` = embedding layer, `-1` = last block. **`None` (default) stores every layer.** Negative indices are resolved to absolute positions before anything touches disk. |
| `cache` | **Required.** Per-layer activations are the stored artefact; there is no in-memory-only path. |
| `pooling` | How to pool `(seq_len, d)` to one vector per query. **Mask-aware**: padded positions are excluded, so a vector depends only on its own query. |
| `normalize_activations` | `"layer"` (default) row-normalizes each layer before concatenating, so every layer weighs the same; `"global"` normalizes only the finished row, letting each layer count in proportion to its own scale; `"none"` leaves it raw. Bools are accepted (`True → "layer"`, `False → "none"`) and canonicalized before hashing. Applies at **read** time — it is part of a surrogate's identity, not of the stored activations. |
| `activation_mode` | `"input"`: forward pass on the prompt. `"generation"`: decoding-step activations, mean-pooled. `"both"`: both stored separately, combined at read time. |
| `max_new_tokens` | Tokens to generate per query; used when `activation_mode` is `"generation"` or `"both"`. Ignored for `"input"`. |
| `view` | Which view `extract()` returns. See below. |

**Views** (assembled at read time from stored per-layer activations, then cached
under `surrogates/`):

| `view` | Shape | Rows are |
|---|---|---|
| `"concat"` (default) | `(n_queries, L·d)` | queries |
| `"gram"` | `(n_queries, n_queries)` | queries |

In `"both"` mode the input and generation halves are concatenated along the
feature axis, so a row stays one query and `L·d` doubles.

**Note on `gram`:** it is a *kernel*, not a feature matrix. `CKADistanceMetric`
forms `K = X Xᵀ` itself, so passing a stored Gram computes `(H Hᵀ)²`. The cache
tags such representations with `metadata["is_kernel"]` and the metric raises on
them. Use `"concat"` with CKA.

**Note on CKA:** `CKADistanceMetric(unbiased=True)` requires ≥ 4 rows and raises
below that rather than returning NaN — the estimator divides by `n(n-3)`. Rows
here are queries, so any realistic query set clears this. Use `unbiased=False`
otherwise. The working note `docs/notes/gram_and_cka.md` covers the two CKA
implementations and how they differ; it is gitignored and not present in a clone.

---

### `LogProbTaxonomy`

```python
class LogProbTaxonomy(HFInferenceTaxonomy, Taxonomy):
    def __init__(
        self,
        queries: Sequence[str],
        query_key: dict | None = None,      # REQUIRED in practice
        cache: LogProbCache | None = None,  # REQUIRED in practice
        device: str = "cuda",
        batch_size: int = 8,                # memory only; not in the cache key
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        mode: str = "input",                # the only accepted value
        max_length: int = 512,              # IS in the cache key
        seq_chunk: int = 64,                # memory only; not in the cache key
        source_indices: list | None = None,
    )
    taxonomy_name = "logprob"
```

Teacher-forces the shared query text through one masked forward pass — the same pass
`FunctionalTaxonomy` runs, with no decoding — and records, at every position, the
log-probability of the token that actually came next and the entropy of the full
next-token distribution.

`extract` returns an `(n_queries, 2)` representation whose columns are the per-query mean
log-probability and mean entropy over the content positions. The full per-token detail is
the stored artifact; read it with `LogProbCache.load_logprobs`.

Both `cache` and `query_key` are required — there is no in-memory-only path, because the
returned representation is a summary of the stored arrays.

`mode="generation"` raises. Generation-mode log-probs are collected by
`BehavioralTaxonomy(collect_logprobs=True, logprob_cache=...)`, because the distributions
they need exist only inside the `generate()` call. Both write into `07_logprobs`.

See [Log-Probability Taxonomy](guides/logprob_taxonomy.md).

---

### `StructuralTaxonomy`

```python
class StructuralTaxonomy(Taxonomy):
    def __init__(
        self,
        layer_names: list[str] | None = None,     # explicit module-name prefixes
        layer_indices: int | list[int] | Literal["last"] | None = None,
        projections: str | list[str] | None = None,   # "k"/"q"/"v"/"o" or list
        lora_only: bool = True,                   # use LoRA adapter matrices only
        use_lora_product: bool = True,            # True = store B@A; False = concat(A, B)
        cache: DiskCache | None = None,           # flat hash-keyed fallback
        lora_cache: LoRACache | None = None,      # hierarchical base_model→adapter cache
        base_model_id: str | None = None,         # auto-detected from PEFT if None
        hf_token: str | None = None,
    )
    taxonomy_name = "structural"
```

| Parameter | Description |
|---|---|
| `layer_names` | Explicit module-name prefixes; takes precedence over `layer_indices` / `projections` when set |
| `layer_indices` | Shorthand layer selector: int, list of ints, `"last"`, or `None` for all layers |
| `projections` | Shorthand projection selector: `"k"`, `"q"`, `"v"`, `"o"` (or long forms), a list, or `None` for all |
| `lora_only` | `True` (default): use only `.lora_A.` / `.lora_B.` parameters; raises `ValueError` if none found |
| `use_lora_product` | `True`: compare `(B @ A).flatten()`; `False`: compare `concat(A.flatten(), B.flatten())` |
| `lora_cache` | `LoRACache` for hierarchical storage under `base_model → adapter → config_hash` |
| `base_model_id` | Base model HF ID; if `None`, read from PEFT `adapter_config.json` on the Hub |

**Extraction priority:** `LoRACache` hit → `DiskCache` hit → local PEFT `adapter_model.safetensors` (no base model load) → full `AutoModelForCausalLM.from_pretrained`.

**Representation shape:** `(N_layers, max_len)` — one row per weight layer or LoRA module. Each vector is stored at its full natural length; rows shorter than the longest are zero-padded so the matrix can be stacked, and the true pre-padding lengths are kept in `metadata["layer_lengths"]` and in the cache `config.json`.

> There is no `n_components` parameter. Vectors were truncated/padded to a fixed width in earlier versions; that was removed so no weight information is discarded.

---

### `DatasetEmbeddingTaxonomy`

```python
from src import DatasetEmbeddingTaxonomy

class DatasetEmbeddingTaxonomy(Taxonomy):
    def __init__(
        self,
        embedder: Embedder,
        datasets: dict[str, tuple[DatasetRecipe | ClassAwareDatasetRecipe, int] |
                              tuple[DatasetRecipe | ClassAwareDatasetRecipe, int, int]],
        representation: Literal["matrix", "gram", "mean"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
        sample_cache: SampledDatasetCache | None = None,
    )

    @classmethod
    def from_recipes(
        cls,
        recipes: list[DatasetRecipe | ClassAwareDatasetRecipe],
        n_samples: int,
        embedder: Embedder,
        representation: Literal["matrix", "gram", "mean"] = "matrix",
        cache: DatasetEmbeddingCache | None = None,
        seed: int = 42,
        hf_token: str | None = None,
        sample_cache: SampledDatasetCache | None = None,
    ) -> DatasetEmbeddingTaxonomy

    def recipe_ids(self) -> list[str]
    taxonomy_name = "dataset_embedding"
```

Compares datasets by embedding their text elements with an `Embedder`. Pass `taxonomy.recipe_ids()` as the `model_ids` argument to `TaxonomyAnalyzer.fit()` — the taxonomy uses recipe hashes as model IDs.

| Parameter | Description |
|---|---|
| `embedder` | Embedder to apply to each text element. Use `SentenceTransformerEmbedder(use_generated_text=False)` to embed dataset text directly. |
| `datasets` | Mapping from recipe ID → `(recipe, n_samples)` or `(recipe, n_samples, seed)`. The optional third element overrides `seed` for that dataset. |
| `representation` | `"matrix"`: raw `(N, d)` embedding matrix. `"gram"`: upper-triangle of `E @ E.T`, shape `(1, N*(N+1)//2)`. `"mean"`: column mean, shape `(1, d)`. |
| `seed` | Global fallback seed for dataset shuffling. |

`from_recipes` is a convenience constructor that registers each recipe under its `recipe_hash()`. Note that the hash is content-addressed, so it does *not* distinguish n or seed: if you register several draws of one mixture this way they collide on the same key. Use explicit recipe IDs (as `make_dataset_embedding_taxonomy` does, keying on the config-block name) whenever n or seed varies.

---

## Dataset recipes

A recipe is a weighted mixture of HuggingFace datasets. `DatasetRecipe` mixes at the
dataset level; `ClassAwareDatasetRecipe` also weights classes within each dataset.
Both serialize to the same `.recipe.json` shape and are content-addressed by
`recipe_hash()`.

```python
from src.datasets.recipe import DatasetRecipe, DatasetEntry
from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
```

| Entry field | Applies to | Description |
|---|---|---|
| `dataset_id` | both | HuggingFace dataset ID. |
| `subset` / `split` | both | Config name and split. `split` defaults to `"train"`. |
| `weight` | both | Relative weight of this dataset in the mixture; normalized across entries. |
| `text_field` | both | **Single-column projection.** Name of the column the entry's text is read from. Default `"text"`. **Ignored entirely when `text_fields` is set** — see below. |
| `text_fields` | both | **Multi-column projection.** List of column names composed into one string. `None` (the default) means use `text_field`. |
| `text_separator` | both | What joins composed columns. Default `"\n"`. Only meaningful with `text_fields`. |
| `class_field` | class-aware | Column whose values distinguish classes. Default `"label"`. |
| `class_filter` | class-aware | Restrict sampling to these class values. |
| `class_weights` | class-aware | Per-class proportions; normalized into `normalized_class_weights`. Uniform over `class_filter` when omitted. |

### `text_field` vs `text_fields`

These are two spellings of one thing — how a dataset row becomes the single string
that gets trained on or embedded — and **only one of them is ever live**.
`src/datasets/_text_projection.py::resolve_text` reads `text_field` *only* when
`text_fields` is empty or absent:

```python
text_field="best_answer"                                   # → "42 is the answer."
text_fields=["question_title", "best_answer"]              # → "What is 6*7?\n42 is the answer."
text_fields=[...], text_field="best_answer"                # → composed; text_field unused
```

Every consumer routes through `entry_text` / `row_text` in that module — the
sampler, `scripts/finetune_lora.py`, and the query builder — so they cannot
disagree about what a row means. `SFTTrainer` takes one column name, so
`finetune_lora.py` materializes a composed entry into a synthetic `_composed_text`
column and passes that as `dataset_text_field`. Composed rows are never written back
into `01_datasets`, which stores source indices only; the composition is a
projection of a draw, not a different draw.

**A recipe on disk may legitimately carry both keys, and that is not a conflict.**
`recipe_hash` is a SHA-256 over `to_dict()` output, so emitting the composition keys
unconditionally would have moved every pre-existing hash and orphaned the draws,
adapters and representations keyed on them. `composition_dict()` therefore omits
them when unset, which leaves `text_field` serialized at whatever value it had —
frequently a stale one, as in the `yahoo_qa_*` recipes where it still reads
`best_answer` while the live projection is `["question_title", "best_answer"]`.
Read `text_fields` first; treat `text_field` as meaningful only in its absence.

**The composition is part of recipe identity.** Two adapters fit on the same rows
projected two different ways are not the same adapter, so composing changes
`recipe_hash` and the composed mixture gets its own directory and its own draw.
`scripts/check_analysis.py` pins the known uncomposed hashes so an accidental
unconditional emit cannot regress silently.

**Missing columns are skipped, not stringified.** A row whose `best_answer` is
absent composes to its question alone rather than to the question followed by the
literal word `None`. A row missing every named column yields `""`, matching the
single-field path.

**Keep the training and probing projections in the same shape.** An adapter fit on
bare answer prose and then prompted with a question is being measured out of
distribution — the failure that motivated `text_fields` in the first place. The
default separator is a bare newline rather than a template marker for exactly this
reason: a richer marker (`### Answer:`, a chat template) has to be mirrored on the
extraction side or it reintroduces the mismatch one level up. See
[Designing probe sets](guides/behavioral_taxonomy.md#designing-probe-sets).

---

## Embedders

Both embedders share this interface:

```python
class Embedder(ABC):
    @abstractmethod
    def embed(self, model_output: Any, probe: str) -> np.ndarray: ...
    @property
    @abstractmethod
    def embedding_dim(self) -> int | None: ...
    @abstractmethod
    def config_dict(self) -> dict[str, Any]: ...
```

### `HiddenStateEmbedder`

```python
class HiddenStateEmbedder(Embedder):
    def __init__(
        self,
        strategy: Literal["hidden_states", "logits"] = "hidden_states",
        layer_index: int = -1,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
    )
```

Extracts vectors from the model's own hidden states or logits. **Only compatible with `FunctionalTaxonomy`** (which passes hidden states to the embedder via the `_InferenceOutput` object). Passing this embedder to `BehavioralTaxonomy` will raise a `ValueError` because `BehavioralTaxonomy` does not collect hidden states.

| Parameter | Description |
|---|---|
| `strategy` | `"hidden_states"` — use transformer hidden states; `"logits"` — use logit vectors |
| `layer_index` | Which layer to extract from; `-1` = last, `-2` = second-to-last, etc. |
| `pooling` | How to aggregate over the sequence dimension |

### `SentenceTransformerEmbedder`

```python
class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        use_generated_text: bool = True,
        normalize_embeddings: bool = True,
    )
```

Encodes text with a sentence-transformers model on CPU. The intended embedder for `BehavioralTaxonomy`. When `use_generated_text=True`, embeds the generated continuation; when `False`, embeds the raw probe string.

---

## Distance metrics

All metrics share this interface:

```python
class DistanceMetric(ABC):
    @abstractmethod
    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float: ...
    @property
    @abstractmethod
    def metric_name(self) -> str: ...
```

### `FrobeniusDistanceMetric`

```python
class FrobeniusDistanceMetric(DistanceMetric):
    def __init__(self, normalize: bool = True)
    metric_name = "frobenius"
```

`normalize=True`: L2-normalizes each row before computing `‖A − B‖_F / √N`. Makes the distance invariant to embedding scale.

### `CKADistanceMetric`

```python
class CKADistanceMetric(DistanceMetric):
    def __init__(
        self,
        kernel: Literal["linear", "rbf"] = "linear",
        sigma: float | None = None,     # RBF bandwidth; None = median heuristic
        unbiased: bool = True,
    )
    metric_name = "cka_linear" | "cka_rbf"
```

Distance = `1 − CKA(A, B)`. Invariant to orthogonal transformations and isotropic scaling. `unbiased=True` requires the matrix row count to be ≥ 4 (rows = `N_layers` for functional, or `2*N_layers` in `"both"` mode).

### `CosineDistanceMetric`

```python
from src.metrics.vector import CosineDistanceMetric

class CosineDistanceMetric(DistanceMetric):
    metric_name = "cosine"
```

Distance = `1 − cosine_similarity(a.flatten(), b.flatten())`. Scale-invariant; both matrices are flattened to 1-D before comparison. Natural companion to `DatasetEmbeddingTaxonomy(representation="mean")`.

### `DotProductDistanceMetric`

```python
from src.metrics.vector import DotProductDistanceMetric

class DotProductDistanceMetric(DistanceMetric):
    metric_name = "dot_product"
```

Distance = `1 − dot(a.flatten(), b.flatten())`. Assumes pre-normalized embeddings (e.g. `SentenceTransformerEmbedder(normalize_embeddings=True)`). For unit vectors this is equivalent to cosine distance.

### `BuresWassersteinDistanceMetric`

```python
class BuresWassersteinDistanceMetric(DistanceMetric):
    metric_name = "bures_wasserstein"
```

The 2-Wasserstein distance between two zero-mean Gaussians with the uncentered
covariances `Σ = XᵀX`, computed without ever forming a `d×d` covariance or a matrix
square root: `d²(A, B) = ‖A‖_F² + ‖B‖_F² − 2‖A Bᵀ‖_*`.

**The two inputs need not have the same number of rows, and row order does not matter** —
`Σ = XᵀX` is invariant to permuting the rows of `X`. It is rank-1 on a single-row
representation and so carries nothing cosine does not; and because it stacks per-block
factors before its SVD, every block must share an input dimension.

### `EnergyDistanceMetric`

```python
from src.metrics import EnergyDistanceMetric

class EnergyDistanceMetric(DistanceMetric):
    metric_name = "energy"
```

`E(A, B)² = 2·E‖x − y‖ − E‖x − x'‖ − E‖y − y'‖`, estimated by U-statistics with self-pairs
excluded. Non-negative, and zero exactly when the two distributions coincide, so the
reported square root is a true metric on distributions.

No bandwidth and no kernel choice, which is the reason to prefer it as the default
distributional reading. It is equivalent to MMD with the Euclidean-distance kernel, so it
and `MMDDistanceMetric` are the same family, **not independent evidence**. Scale-sensitive
by construction.

### `MMDDistanceMetric`

```python
from src.metrics import MMDDistanceMetric

class MMDDistanceMetric(DistanceMetric):
    def __init__(
        self,
        kernel: Literal["rbf", "linear"] = "rbf",
        sigma: float | None = None,     # None = median heuristic over the pooled pair
    )
    metric_name = "mmd_rbf" | "mmd_linear"
```

Reported as `sqrt(max(MMD², 0))` using the unbiased estimator. **The clamp is not
cosmetic**: under the null the unbiased MMD² is centred on zero and negative about half
the time, so a value of exactly `0.0` should be read as *at or below the noise floor*,
not as *identical*. The biased estimator would avoid the clamp only by putting a floor
under every distance that grows as row counts shrink — worse, because it varies across
the very models being compared.

`kernel="rbf"` is characteristic, so MMD = 0 implies the distributions are equal.
`kernel="linear"` sees only the difference of means and is the explicit degenerate case —
what a cosine or Frobenius comparison of pooled centroids already measures.

With `sigma=None` the bandwidth is recomputed per pair, because one derived from a single
model's scale would make `d(a, b) != d(b, a)`. The consequence: the metric stays symmetric
but is **not a metric across the collection**, since different pairs are measured with
different kernels. **Pass an explicit `sigma` when a distance matrix must be internally
consistent** — an MDS embedding of one is exactly that case.

> `EnergyDistanceMetric` and `MMDDistanceMetric` both require more than one row and reject
> kernel-matrix representations. See
> [Cross-Level Comparison](guides/cross_level_comparison.md#distributional-distance-metrics).

---

## Geometry methods

All geometry methods share this interface:

```python
class GeometryMethod(ABC):
    @abstractmethod
    def fit(self, distance_matrix: DistanceMatrix) -> GeometryResult: ...
    @property
    @abstractmethod
    def method_name(self) -> str: ...
```

### `MDSGeometry`

```python
class MDSGeometry(GeometryMethod):
    def __init__(
        self,
        n_components: int = 2,
        metric: bool = True,
        max_iter: int = 300,
        n_init: int = 4,
        random_state: int = 0,
    )
    method_name = "mds"
```

Returns `GeometryResult.stress` (lower is better; < 0.10 is good).

### `PCAGeometry`

```python
class PCAGeometry(GeometryMethod):
    def __init__(self, n_components: int = 2)
    method_name = "pca"
```

Deterministic. `geometry.metadata["explained_variance_ratio"]` contains the fraction of variance explained by each component.

### `UMAPGeometry`

```python
class UMAPGeometry(GeometryMethod):
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 5,
        min_dist: float = 0.1,
        random_state: int = 0,
    )
    method_name = "umap"
```

Requires `umap-learn`. `n_neighbors` is automatically clamped to `N − 1`.

---

## Compute backends

All backends share this interface:

```python
class ComputeBackend(ABC):
    @abstractmethod
    def map_extract(
        self, taxonomy: Taxonomy, model_ids: Sequence[ModelID]
    ) -> list[ModelRepresentation]: ...

    @abstractmethod
    def map_distances(
        self, metric: DistanceMetric, representations: Sequence[ModelRepresentation]
    ) -> np.ndarray: ...
```

### `LocalBackend`

```python
class LocalBackend(ComputeBackend):
    def __init__(self, n_jobs: int = 1)
```

`n_jobs=1` runs sequentially (required for GPU models). `n_jobs=-1` uses all CPU cores (safe for distance computation; safe for CPU-only models).

### `SlurmBackend`

```python
class SlurmBackend(ComputeBackend):
    def __init__(
        self,
        slurm_params: dict,
        results_dir: Path = Path("./slurm_jobs"),
        n_distance_jobs: int = 1,
    )
```

`slurm_params` is passed directly to `submitit.AutoExecutor.update_parameters()`. One SLURM job per model. `n_distance_jobs` controls parallelism for the local distance computation step.

---

## Cache

### `DiskCache`

```python
class DiskCache:
    def __init__(
        self,
        cache_dir: Path | str,
        format: Literal["npz", "pt", "safetensors"] = "safetensors",
    )

    def exists(self, key: str) -> bool
    def load(self, key: str) -> ModelRepresentation
    def save(self, key: str, rep: ModelRepresentation) -> None
    @staticmethod
    def key_for(model_id: ModelID, config: dict) -> str
```

| Format | Description |
|---|---|
| `"safetensors"` (default) | Memory-mappable, pickle-free, fast load. Recommended. |
| `"npz"` | NumPy zip archive. Portable backward-compatible option. |
| `"pt"` | PyTorch format. Preserves `bfloat16` precision. |

Files are stored at `cache_dir/{key[:2]}/{key}.{ext}`. Writes are atomic (`os.replace`) and protected by a per-key `filelock`, making concurrent SLURM writes safe.

---

### `LoRACache`

```python
class LoRACache:
    def __init__(self, cache_root: Path | str)

    def exists(self, base_model_id: str, adapter_id: str) -> bool
    def save(
        self,
        base_model_id: str,
        adapter_id: str,
        rep: ModelRepresentation,
        training_config: dict,
        extraction_config: dict,
        dataset_recipe: dict | None = None,
    ) -> None
    def load(self, base_model_id: str, adapter_id: str) -> ModelRepresentation
    def load_config(self, base_model_id: str, adapter_id: str) -> dict
    def list_adapters(self, base_model_id: str) -> list[str]
    def list_base_models(self) -> list[str]

    @staticmethod
    def detect_base_model(adapter_id: str, hf_token: str | None = None) -> str
    @staticmethod
    def _read_peft_adapter_config(adapter_id: str, hf_token: str | None = None) -> dict
```

Hierarchical cache for structural (LoRA) representations. Stores a `config.json` (with `training_config` and a `dataset_recipe` stub) and `representation.safetensors` per adapter under `base_model → adapter` directories.

`detect_base_model` reads the PEFT `adapter_config.json` from the Hub to resolve the base model ID automatically.

---

### `CollectionCache`

```python
class CollectionCache:
    def __init__(self, cache_root: Path | str)

    # The handle: "{taxonomy}/{collection_key}/{metric}_{surrogate_key}"
    @staticmethod
    def collection_key(model_entries: list[dict]) -> str
    @staticmethod
    def surrogate_key(surrogate_hashes: list[str | None]) -> str
    @staticmethod
    def handle(taxonomy: str, collection_key: str, metric: str,
               surrogate_key: str) -> str

    def exists(self, handle: str) -> bool
    def save_distance_matrix(
        self,
        distance_matrix: DistanceMatrix,
        handle: str,
        model_entries: list[dict] | None = None,
        label: str | None = None,
        slice_key: dict | None = None,
        config: dict | None = None,
    ) -> str    # returns the handle

    def save_geometry(self, handle: str, geometry: GeometryResult,
                      mds_kwargs: dict | None = None) -> None
    def load_distance_matrix(self, handle: str) -> DistanceMatrix
    def load_geometry(self, handle: str, method: str,
                      n_components: int | None = None,
                      mds_kwargs: dict | None = None) -> GeometryResult
    def load_info(self, handle: str) -> dict      # collection level
    def load_config(self, handle: str) -> dict    # leaf: spec + per-model hashes
    def list_collections(self) -> list[str]
    def find(self, **criteria) -> list[str]
```

Stores distance matrices and geometry results for a model collection.

**A collection is keyed on what it was built from, not on its model list.** The
handle has three parts: `collection_key` hashes each model's stored artifact
path (relative to the cache root, stopping before `surrogates/`), and
`surrogate_key` hashes the ordered list of per-model surrogate hashes — the
read-time view. So two comparisons that differ in draw, embedder, view,
normalization or pooling get different directories, while two that read the same
tensors share one even if the selector was spelled differently.

This replaced a key of `(sorted model_ids, taxonomy, metric)`, which was blind to
all of the above and silently returned one selector's matrix for another. See
`docs/notes/TODO.md` item 14. Entries written under the old key are not readable
under the new one; `scripts/migrate_collection_key.py` quarantines them.

`collection_info.json` sits at the `collection_key` level and records the models
and their artifact paths, shared by every metric and view computed over them.
Each leaf's `config.json` records the surrogate spec and every per-model
surrogate hash — since the directory name is a digest, that file is what traces a
collection back to its inputs. `save_geometry` can be called repeatedly to add
embeddings at several dimensions to the same leaf.

`model_entries` is an ordered list of dicts describing each model in `distance_matrix.model_ids`. Each entry should have at minimum `{"model_id": ..., "entry_type": "base_model" | "lora_adapter"}`. LoRA adapter entries can additionally record `base_model_id` and `adapter_cache_slug` to allow cache lookup.

---

### `DatasetEmbeddingCache`

```python
from src import DatasetEmbeddingCache

class DatasetEmbeddingCache:
    def __init__(self, cache_root: Path | str)

    def exists(self, recipe_hash: str, n_samples: int, seed: int,
               embedder_hash: str, spec: dict) -> bool
    def save(
        self,
        recipe: DatasetRecipe | ClassAwareDatasetRecipe,
        rep: ModelRepresentation,
        embedder_config: dict,
        representation: str,
        n_samples: int,
        seed: int,
    ) -> None
    def load(self, recipe_hash: str, n_samples: int, seed: int,
             embedder_hash: str, spec: dict) -> ModelRepresentation

    def list_draws(self, recipe_hash: str) -> list[tuple[int, int]]
    def list_embedder_hashes(self, recipe_hash: str, n_samples: int, seed: int) -> list[str]
    def list_surrogates(self, recipe_hash: str, n_samples: int, seed: int,
                        embedder_hash: str) -> list[dict]

    @staticmethod
    def embedder_hash(embedder_config: dict) -> str
    @staticmethod
    def surrogate_hash(spec: dict) -> str
    @staticmethod
    def spec_for(representation: str) -> dict
```

Hierarchical cache for `DatasetEmbeddingTaxonomy` representations, stored under
`cache_root/02_dataset_embeddings/{recipe_hash}/n{n}_s{seed}/{embedder_hash}/`, with
a human-readable `recipe.json` at the recipe level, a `config.json` naming the
embedder at the entry level, and the tensor under
`surrogates/{surrogate_hash}/surrogate.safetensors`. Pass an instance to
`DatasetEmbeddingTaxonomy(cache=...)` to enable persistence.

`embedder_hash` identifies the embedder configuration and nothing else. It used to
carry `representation`, `n_samples` and `seed` as well; the draw is now a path
component and the representation is the surrogate spec, so the key finally means
what its name says. `seed` is required by `save` — it names a directory, and `None`
would render as the literal `sNone`.

`surrogate_hash` is deliberately `DrawKeyedCache.config_hash`, so a spec dict hashes
identically here and at `04`/`05`.

**A surrogate here is authored, not derived.** At `04`/`05` a surrogate is a
read-time view of a stored base artifact; this stage stores no base, because the
full `(N, 768)` embeddings would cost 6.1 GB and a GPU re-embed. Adding a
representation means re-embedding, not a read-time rebuild.

---

### `LogProbCache`

```python
from src.cache.logprob_cache import LogProbCache

class LogProbCache(DrawKeyedCache):
    _STAGE_DIR = "07_logprobs"

    def exists(base_model_id, adapter_id, query_key, mode, *,
               max_new_tokens=None, replicates=None, sampling_hash=None) -> bool
    def save_logprobs(base_model_id, adapter_id, query_key, mode, arrays, *,
                      max_new_tokens=None, replicates=None, sampling=None,
                      model_id=None, config=None, run_metadata=None,
                      source_indices=None) -> None
    def load_logprobs(base_model_id, adapter_id, query_key, mode, *,
                      max_new_tokens=None, replicates=None,
                      sampling_hash=None) -> tuple[dict[str, np.ndarray], dict]
    def list_entries(base_model_id, adapter_id, query_key) -> list[str]

    @staticmethod
    def masked_mean(values, lengths, start=None) -> np.ndarray
```

`save_logprobs` is **idempotent on filename**, like every other stage here — which is safe
only because every axis that changes the numbers is in the filename. `artifact_name`
reuses `GeneratedTextCache.variant_token` as the same function object, so a log-prob file
carries the same token as the generation it describes.

`load_logprobs` returns `(arrays, meta)`. `masked_mean(values, lengths, start=None)` is the
one reduction every reader of this stage needs, kept here so the padding convention is
applied in one place; `start` trims leading scaffolding (`content_start`).

`list_entries` is a directory listing with no file opens — the peer of
`GeneratedTextCache.list_variants`.

---

## Analysis

Import from `src.analysis` (also re-exported from `src`). Everything here
consumes `DistanceMatrix` / `GeometryResult`; nothing loads a model or touches a
GPU.

### Bridge — raw LoRA weights → core types

```python
as_distance_matrix(names, matrix, metric, taxonomy="structural",
                   similarity=False) -> DistanceMatrix
lora_distance_matrix(weights, kind="cosine", layers=None, projections=None,
                     align=False, cache_dir=None) -> DistanceMatrix
fit_geometry(dm, method="mds", n_components=2, **kwargs) -> GeometryResult
save_collection(dm, geometries=(), cache_root=..., model_entries=None) -> str
```

`kind` is one of `"cosine"`, `"frobenius"`, `"bures_wasserstein"`, `"cka"`.
`similarity=True` converts `1 - S`, needed for `cosine_similarity_matrix`.
`"cka"` accepts a single `(layer, projection)` block only.

### Distance-matrix comparison

```python
match_models(*objs, key=None) -> tuple[list[ModelID], list[np.ndarray]]
offdiag(matrix) -> np.ndarray
matrix_correlation(dm_a, dm_b, method="spearman", key=None) -> float
mantel_test(dm_a, dm_b, n_permutations=9999, method="spearman",
            random_state=0, key=None) -> MantelResult
distance_correlation(dm_a, dm_b, bias_corrected=True, key=None) -> float
dcor_test(dm_a, dm_b, n_permutations=9999, bias_corrected=True,
          random_state=0, key=None) -> DcorResult
correlation_table(analyses, method="spearman", min_models=3, key=None)
    -> tuple[list[str], np.ndarray]
```

`match_models` is set intersection plus reindexing — bookkeeping, unrelated to
Procrustes. `correlation_table` reports `nan` for pairs with no models in common
rather than raising.

`MantelResult`: `statistic`, `p_value`, `n_permutations`, `n_models`, `method`, `null`.
Quote the statistic; **do not quote the p-value** — the off-diagonal entries are
dependent and it is not calibrated. Use `dcor_test` or `protest` for inference.

`DcorResult`: `statistic`, `p_value`, `n_permutations`, `n_models`,
`bias_corrected`, `exact`, `null`.

`distance_correlation` defaults to the bias-corrected `dCor*` of Székely & Rizzo
(2013), which lives on the squared scale and **may be negative**. The default is
not cosmetic: between two *independent* five-model matrices the classical
statistic averages 0.85, so it cannot distinguish signal from nothing at the
sizes here. `dcor_test` enumerates all `n!` relabellings when `n! <=
n_permutations` and sets `exact`.

Two limits that decide how to read the output:

- **dCor is unsigned.** A taxonomy that recovers the ordering exactly backwards
  scores 1.0, same as a perfect one. Always report `matrix_correlation` beside it
  — though note that at the matrix level the signed correlation is blind to
  reversal too. Only the recovery correlation, downstream of the barycentric
  projection, sees direction.
- **The reference matrix's symmetry floors the p-value.** Every relabelling that
  maps the U-centred reference onto itself contributes a tied null value, so
  `p >= #Aut(B) / n!` for any input. The ground truth on these slices — five
  evenly-spaced mixtures — has 8 such relabellings, so an exact recovery scores
  `p = 8/120 ≈ 0.067` and nothing on that slice reaches `p < 0.05`. It is a
  property of the design, not a weak result, and not a universal `n = 5` limit:
  an unevenly-spaced truth has 4 and reaches `0.033`. Rank by the statistic.

### Identifier reconciliation

```python
recipe_id_for(model_id) -> str
relabel(obj, key) -> DistanceMatrix | GeometryResult | dict
id_overlap(*objs, key=None) -> dict
```

Model-level taxonomies key rows by adapter path
(`.../yahoo_topic0_only_r16`); `DatasetEmbeddingTaxonomy` keys them by recipe ID
(`yahoo_topic0_only`). `recipe_id_for` maps the former onto the latter, reading
`dataset_name` from the adapter's `experiment_meta.json` and falling back to
parsing the directory name; non-adapter identifiers pass through unchanged. Pass
it as `key=` to any comparison function to bring `dataset_embedding` into a
cross-taxonomy table.

`relabel` returns a copy with rewritten `model_ids` (arrays shared, input
untouched) and accepts a callable or a partial mapping. It raises if the rewrite
would collide two distinct models — as a LoRA rank or init-seed sweep would,
since those variants share one dataset. `id_overlap` reports identifier counts,
intersection size and the entries unique to each side, with or without a `key`.

### Configuration comparison

```python
procrustes_compare(geom_a, geom_b, scaling=True, reflection=True) -> ProcrustesResult
per_point_residuals(result) -> np.ndarray
protest(geom_a, geom_b, n_permutations=9999, random_state=0, ...) -> ProtestResult
align_to_reference(geometries, reference=None, ...) -> list[GeometryResult]
point_dispersion(geometries, reference=None, ...) -> DispersionResult
```

`ProcrustesResult`: `disparity`, `aligned_a`, `aligned_b`, `rotation`, `scale`,
`model_ids`. With `scaling=True` the disparity matches
`scipy.spatial.procrustes` and lies in `[0, 1]`.

`DispersionResult`: `per_model`, `model_ids`, `mean_disparity`, `n_geometries`,
`sorted_models()`.

### Embedding fidelity

```python
kruskal_stress(dm, geometry) -> float
shepard(dm, geometry) -> tuple[np.ndarray, np.ndarray]
```

### Anchor simplex

```python
barycentric(geometry, anchors, clip=True) -> SimplexProjection
compare_simplices(a, b) -> SimplexComparison
anchor_weight_vs_truth(projection, true_values, anchor=0) -> RecoveryResult
```

```python
@dataclass
class SimplexProjection:
    weights:    np.ndarray   # (n, k), rows sum to 1
    model_ids:  list[ModelID]
    anchor_ids: list[ModelID]
    residuals:  np.ndarray   # (n,) distance from the simplex / its affine hull
    taxonomy:   str
    method:     str
    clipped:    bool
```

with `weight_for()`, `anchor_column()`, `save()` and `load()`.

`SimplexComparison`: `l1`, `total_variation`, `l2`, `per_anchor_spearman`,
`mean_total_variation`, `max_total_variation`, `sorted_models()`.

`RecoveryResult`: `true`, `recovered`, `pearson`, `spearman`, `residuals`,
`columns`, `mean_l1`, plus `.r` / `.rho` / `.pairs()` for single-column
comparisons. `true_values` accepts a mapping keyed by model ID or an array of
shape `(n,)` or `(n, k)`.

See [Core Concepts](concepts.md) for why barycentric coordinates make two
geometries comparable without a Procrustes step.

---

### Fleet-level surrogate transforms

```python
from src.analysis.surrogates import (
    center_representations, whiten_representations,
    centered, whitened, transform_key,
)

center_representations(reps, mode="grand" | "rowwise") -> list[ModelRepresentation]
whiten_representations(reps, shrinkage=0.1, mode="grand") -> list[ModelRepresentation]

centered(mode="grand") -> Transform
whitened(shrinkage=0.1, mode="grand") -> Transform
transform_key(transform) -> str
```

Collection-level operations a `DistanceMetric` cannot perform, applied between resolution
and distancing. `centered` / `whitened` return the callable that
`build_taxonomy_artifacts(..., transform=)` and `resolve_ordered(..., transform=)` accept.
Results are fresh objects tagged in `metadata["surrogate_transform"]`.

### Scoring a taxonomy against the ground truth

```python
from src.analysis import dcor_vs_truth, disparity_vs_truth

dcor_vs_truth(dm: DistanceMatrix, truth_dm: DistanceMatrix) -> float
disparity_vs_truth(dm, truth_geometry, *, geometry=None,
                   random_state=0, n_components=2) -> float
```

`dcor_vs_truth` scores the **distances** and never embeds; higher is better, and the
U-centred dCor\* may legitimately be negative — do not clip it. `disparity_vs_truth`
scores the **configuration** an embedding draws, in `[0, 1]` with **0 meaning identical
shape**, so it runs opposite to `dcor_vs_truth`. Pass `geometry=` to score an embedding
you already fitted, and pass the `random_state` that embedding was fitted under otherwise.

### Collection-level resolution

```python
build_taxonomy_artifacts(index, taxonomy, metric="cosine", ..., transform=None)
    -> tuple[DistanceMatrix, dict[str, GeometryResult]]

resolve_ordered(index, taxonomy, ids, *, layers=None, projections=None,
                embedder_hash=None, dataset_selector=None,
                behavioral_selector=None, functional_selector=None,
                transform=None, with_identity=False)
    -> tuple[list | None, list[int]] | tuple[list | None, list[int], list[dict]]

collection_handle(cache, taxonomy, metric, model_entries, *,
                  transform=None, rung=None) -> str
```

`resolve_ordered` exists so a **sweep over metrics at one selector resolves once** — the
shape of every panel grid. `order` is the permutation into `index.entries`; structural
returns `reps is None` because it reads the adapter files itself. With
`with_identity=True` it also returns one identity dict per requested id, in `ids` order,
carrying the `artifact_path` and `surrogate_hash` a collection is keyed on — so a caller
that wants to cache what it resolved does not have to resolve twice to find out what it
resolved.

`collection_handle` composes `{taxonomy}/{collection_key}/{metric}_{surrogate_key}` in one
place, so every writer into `06_collections` keys identically. The metric's **reported**
name is used (`"cka"` → `"cka_linear"`). `transform` and `rung` are the two things
resolution does not show; both join the surrogate key, tagged and only when present, so a
raw, rung-less handle is unchanged from before either existed. `rung` is the **resolved
selector dict** — two rungs of one level read the same artifacts under the same surrogate,
so without it they collide — and never the row's display label, which is editable prose.

---

## Model profiles and prompt formats

```python
from src.models.profile import ModelProfile, resolve, assert_compatible, template_sha

@dataclass(frozen=True)
class ModelProfile:
    match: str                                    # an id PREFIX; longest match wins
    torch_dtype: str = "float16"
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    prompt_format: str = "raw"                    # "raw" | "chat"
    chat_template_sha: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    notes: str = ""

resolve(model_id) -> ModelProfile        # falls back to a bare default, never raises
assert_compatible(profile, tokenizer)    # raises on template drift or a wrong format
template_sha(tokenizer) -> str | None
```

```python
from src.datasets._chat_projection import (
    PromptFormat, render_prompt, render_pair, encode_pair,
)

@dataclass
class PromptFormat:
    format: str = "raw"                  # "raw" | "chat"
    user_fields: tuple[str, ...] = ()
    answer_fields: tuple[str, ...] = ()
    separator: str = DEFAULT_SEPARATOR
    answer_separator: str = DEFAULT_SEPARATOR
    chat_template_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, block: dict | None) -> PromptFormat
    def to_dict(self) -> dict                # {} when raw
    def format_id(self) -> str | None        # 8 hex chars, or None when raw
```

A profile is a source of **defaults for spec construction** and never a runtime lookup:
the generator resolves one when it emits YAML and writes every value out explicitly.
`encode_pair` asserts that tokenizing prompt and completion separately reproduces the
joint tokenization. See [Model Profiles and Prompt Formats](guides/model_profiles.md).

---

## Experiment suites

```python
from src.experiments import Suite

@dataclass(frozen=True)
class Suite:
    tag: str = ""
    base_model: str = "meta-llama/Llama-3.1-8B"
    torch_dtype: str = "float16"
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    prompt_format: dict = field(default_factory=dict)
    query_sets: tuple[str, ...] = ("full_context", "question_only")
    job_tokens: dict = ...
    emit_logprob_jobs: bool = False
    emit_gen_activation_job: bool = False
    temperature_sweep: tuple[float, ...] = ()
    sweep_replicates: int = 8
    sweep_shards: int = 4
    emit_embed_jobs: bool = True
    emit_build_job: bool = True
    job_prefix: str = "s3"
    # walls, memory, partitions, batch — see the guide

    @property
    def model_slug(self) -> str
    @property
    def suffix(self) -> str
    @property
    def effective_batch(self) -> int
    def job_token(self, query_set: str) -> str
    def for_model(self, base_model: str, **overrides) -> Suite
```

`for_model` pulls dtype, LoRA targets and prompt-format defaults from the model's profile.
`Suite()` must regenerate `experiments/simplex3` and `jobs/simplex3` byte-for-byte. See
[Experiment Suites](guides/experiment_suites.md).

---

## Abstract base classes

Import from `src.core.protocols` to subclass when extending the library:

```python
from src.core.protocols import Taxonomy, Embedder, DistanceMetric, GeometryMethod, ComputeBackend
```

See [Extending the Library](guides/extending.md) for full examples.
