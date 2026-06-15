# API Reference

All public classes are importable directly from `src`:

```python
from src import BehavioralTaxonomy, CKADistanceMetric, MDSGeometry, ...
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
    n_probes: int
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
    def save(self, path: Path) -> None
    @classmethod
    def load(cls, path: Path) -> DistanceMatrix
```

`sorted_neighbors` returns all other models sorted by ascending distance.

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
    def save(self, path: Path) -> None
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

    def save(self, path: Path) -> None
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
class BehavioralTaxonomy(Taxonomy):
    def __init__(
        self,
        probes: Sequence[str],
        embedder: Embedder,
        cache: DiskCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        max_new_tokens: int = 0,         # 0 = no generation
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,     # falls back to HF_TOKEN env var
    )
    taxonomy_name = "behavioral"
```

---

### `FunctionalTaxonomy`

```python
class FunctionalTaxonomy(Taxonomy):
    def __init__(
        self,
        probes: Sequence[str],
        layer_indices: list[int],                      # e.g. [-4, -3, -2, -1]
        cache: DiskCache | None = None,
        device: str = "cuda",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.float16,
        hf_token: str | None = None,
        pooling: Literal["mean", "last_token", "cls"] = "mean",
        normalize_activations: bool = True,
    )
    taxonomy_name = "functional"
```

| Parameter | Description |
|---|---|
| `layer_indices` | Indices into `hidden_states`; `-1` = last transformer block, `0` = embedding layer |
| `pooling` | How to pool the `(seq_len, d)` hidden state to a single vector per probe |
| `normalize_activations` | L2-normalize activation vectors before computing Gram matrix; makes `G[i,i]=1` |

**Representation shape:** `(N_layers, N_probes*(N_probes+1)//2)` — one row per layer, columns are the upper triangle of the Gram matrix `H @ H.T`.

**Note on CKA:** `CKADistanceMetric(unbiased=True)` requires `N_layers ≥ 4`. Use `unbiased=False` for smaller layer sets.

---

### `StructuralTaxonomy`

```python
class StructuralTaxonomy(Taxonomy):
    def __init__(
        self,
        layer_names: list[str] | None = None,   # None = auto-detect
        n_components: int = 256,                 # per-layer vector length after truncate/pad
        lora_only: bool = True,                  # use LoRA adapter matrices only
        use_lora_product: bool = False,          # True = store B@A; False = concat(A, B)
        cache: DiskCache | None = None,
        hf_token: str | None = None,
    )
    taxonomy_name = "structural"
```

| Parameter | Description |
|---|---|
| `layer_names` | Explicit parameter names to compare; `None` = auto-detect LoRA pairs or all 2-D weights |
| `n_components` | Each per-layer weight vector is truncated or zero-padded to this length |
| `lora_only` | `True` (default): use only `.lora_A.` / `.lora_B.` adapter parameters; raises `ValueError` if none found |
| `use_lora_product` | `True`: compare `(B @ A).flatten()` (weight delta); `False`: compare `concat(A.flatten(), B.flatten())` |

**Representation shape:** `(N_layers, n_components)` — one row per weight layer or LoRA module. No input probes required.

**LoRA detection:** Matches parameters whose names contain `.lora_A.` and `.lora_B.`, covering standard PEFT naming. Models with merged LoRA weights will not have these parameters; use `lora_only=False` for them.

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

Requires `max_new_tokens > 0` in `BehavioralTaxonomy` when `use_generated_text=True`.

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

`normalize=True`: L2-normalizes each row before computing `‖A − B‖_F / √N`. This makes the distance invariant to embedding scale.

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

Distance = `1 − CKA(A, B)`. CKA is invariant to orthogonal transformations and isotropic scaling. The unbiased HSIC estimator is more reliable for small probe sets. For the RBF kernel, `sigma=None` uses the median pairwise distance heuristic.

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
        metric: bool = True,      # True = metric MDS, False = non-metric
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

`n_jobs=1` runs sequentially (required for GPU models). `n_jobs=-1` uses all CPU cores (safe for distance computation, safe for CPU-only models).

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

`slurm_params` is passed directly to `submitit.AutoExecutor.update_parameters()`. One SLURM job is submitted per model. `n_distance_jobs` controls parallelism for the local distance computation step.

---

## Cache

### `DiskCache`

```python
class DiskCache:
    def __init__(
        self,
        cache_dir: Path | str,
        format: Literal["npz", "pt"] = "npz",
    )

    def exists(self, key: str) -> bool
    def load(self, key: str) -> ModelRepresentation
    def save(self, key: str, rep: ModelRepresentation) -> None
    @staticmethod
    def key_for(model_id: ModelID, config: dict) -> str
```

`format="npz"` (default) is portable and compatible with any numpy installation. `format="pt"` preserves `bfloat16` precision when models use it.

Files are stored at `cache_dir/{key[:2]}/{key}.{ext}`. Writes are atomic (`os.replace`) and protected by a per-key `filelock`, making concurrent SLURM writes safe.

---

## Abstract base classes

Import from `src.core.protocols` to subclass when extending the library:

```python
from src.core.protocols import Taxonomy, Embedder, DistanceMetric, GeometryMethod, ComputeBackend
```

See [Extending the Library](guides/extending.md) for full examples.
