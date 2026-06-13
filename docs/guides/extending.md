# Extending the Library

The library is designed so that new taxonomies, distance metrics, geometry methods, and compute backends can be added without changing any existing code. Each is an independent subclass of an abstract base class.

## Adding a new taxonomy

To add a new taxonomy (e.g., `StructuralTaxonomy` comparing model weights), subclass `Taxonomy` and implement three methods:

```python
# src/taxonomy/structural.py

from __future__ import annotations
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from src.core.protocols import Taxonomy, ModelID
from src.core.representation import ModelRepresentation
from src.cache.disk import DiskCache


class StructuralTaxonomy(Taxonomy):
    """Compares models by their weight matrix geometry.

    Extracts weight matrices from specified layers and projects them
    into a common representation space.
    """

    def __init__(
        self,
        layer_names: list[str],        # e.g. ["model.layers.0.self_attn.q_proj.weight"]
        n_components: int = 256,       # PCA projection dimension
        cache: DiskCache | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.layer_names = layer_names
        self.n_components = n_components
        self.cache = cache
        self.hf_token = hf_token

    @property
    def taxonomy_name(self) -> str:
        return "structural"

    def config_dict(self) -> dict[str, Any]:
        # All extraction parameters — changing any one invalidates the cache
        return {
            "taxonomy": "structural",
            "layer_names": sorted(self.layer_names),
            "n_components": self.n_components,
        }

    def extract(self, model_id: ModelID) -> ModelRepresentation:
        cache_key = DiskCache.key_for(model_id, self.config_dict()) if self.cache else ""
        if self.cache and self.cache.exists(cache_key):
            return self.cache.load(cache_key)

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            device_map="cpu",             # weights don't need GPU
            token=self.hf_token,
        )

        vectors = []
        for name, param in model.named_parameters():
            if name in self.layer_names:
                w = param.detach().float().numpy()
                # Flatten and truncate/pad to n_components
                v = w.flatten()[:self.n_components]
                if len(v) < self.n_components:
                    v = np.pad(v, (0, self.n_components - len(v)))
                vectors.append(v)

        del model
        matrix = np.stack(vectors, axis=0)  # (n_layers, n_components)

        rep = ModelRepresentation.create(
            model_id=model_id,
            taxonomy=self.taxonomy_name,
            matrix=matrix,
            config=self.config_dict(),
            metadata={"n_layers": len(vectors)},
        )
        if self.cache:
            self.cache.save(cache_key, rep)
        return rep
```

### What to implement

| Method | Required | Description |
|---|---|---|
| `taxonomy_name` | Yes | String identifier; appears in `TaxonomyAnalysis.taxonomy_name` |
| `config_dict()` | Yes | All config that affects the representation; used for cache keying |
| `extract(model_id)` | Yes | Load model, compute matrix, unload model, return `ModelRepresentation` |

### Guidelines

- **Always unload the model.** Call `del model` and `torch.cuda.empty_cache()` (if GPU) before returning. The `TaxonomyAnalyzer` calls `extract()` in sequence; a leaked model will prevent the next one from loading.

- **Use `ModelRepresentation.create()`** rather than the raw constructor. It computes the `cache_key` automatically from your `config_dict()`.

- **Matrix rows are the probe/sample dimension.** For the behavioral taxonomy, rows are probe inputs. For a structural taxonomy where each row is a weight layer, rows are layers. The `DistanceMetric` operates on the full matrix and must receive consistent shapes across all models.

- **Make it pickle-safe.** If you plan to use `SlurmBackend`, the taxonomy instance must be serializable. Avoid storing open file handles, live tensors, or lambda functions as instance attributes.

---

## Adding a new distance metric

Subclass `DistanceMetric` and implement `compute()`:

```python
# src/metrics/rsa.py

import numpy as np
from scipy.stats import spearmanr

from src.core.protocols import DistanceMetric
from src.core.representation import ModelRepresentation


class RSADistanceMetric(DistanceMetric):
    """Representational Similarity Analysis distance.

    Computes the pairwise distance matrices within each representation,
    then measures Spearman correlation between those distance vectors.
    Distance = 1 - correlation.
    """

    @property
    def metric_name(self) -> str:
        return "rsa"

    def compute(self, a: ModelRepresentation, b: ModelRepresentation) -> float:
        from scipy.spatial.distance import pdist

        # Pairwise distances within each representation (probe-to-probe)
        rdm_a = pdist(a.matrix, metric="correlation")
        rdm_b = pdist(b.matrix, metric="correlation")

        rho, _ = spearmanr(rdm_a, rdm_b)
        return float(1.0 - rho)
```

The `pairwise()` loop and `DistanceMatrix` construction are handled by the `ComputeBackend` — you only need to implement the scalar `compute()` for a single pair.

---

## Adding a new geometry method

Subclass `GeometryMethod` and implement `fit()`:

```python
# src/geometry_methods/tsne.py

import numpy as np
from sklearn.manifold import TSNE

from src.core.protocols import GeometryMethod
from src.core.distance import DistanceMatrix
from src.core.geometry import GeometryResult


class TSNEGeometry(GeometryMethod):
    """t-SNE embedding from a precomputed distance matrix."""

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 5.0,
        random_state: int = 0,
    ) -> None:
        self.n_components = n_components
        self.perplexity = perplexity
        self.random_state = random_state

    @property
    def method_name(self) -> str:
        return "tsne"

    def fit(self, distance_matrix: DistanceMatrix) -> GeometryResult:
        tsne = TSNE(
            n_components=self.n_components,
            metric="precomputed",
            perplexity=min(self.perplexity, len(distance_matrix.model_ids) - 1),
            random_state=self.random_state,
        )
        coords = tsne.fit_transform(distance_matrix.matrix)

        return GeometryResult(
            coordinates=coords.astype(np.float32),
            model_ids=distance_matrix.model_ids,
            method=self.method_name,
            taxonomy=distance_matrix.taxonomy,
            n_components=self.n_components,
            stress=None,
        )
```

---

## Adding a new compute backend

Subclass `ComputeBackend` and implement `map_extract()` and `map_distances()`:

```python
# src/compute/ray_backend.py

from typing import Sequence
import numpy as np

from src.core.protocols import Taxonomy, DistanceMetric, ComputeBackend, ModelID
from src.core.representation import ModelRepresentation


class RayBackend(ComputeBackend):
    """Compute backend using Ray for distributed execution."""

    def __init__(self, num_cpus: int | None = None) -> None:
        self.num_cpus = num_cpus

    def map_extract(
        self,
        taxonomy: Taxonomy,
        model_ids: Sequence[ModelID],
    ) -> list[ModelRepresentation]:
        import ray

        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus)

        extract_remote = ray.remote(taxonomy.extract)
        futures = [extract_remote.remote(mid) for mid in model_ids]
        return ray.get(futures)

    def map_distances(
        self,
        metric: DistanceMetric,
        representations: Sequence[ModelRepresentation],
    ) -> np.ndarray:
        from src.compute.local import LocalBackend
        return LocalBackend(n_jobs=-1).map_distances(metric, representations)
```

---

## Registering new components

There is no registry to update — new classes are imported directly. Add your new class to the relevant `__init__.py` for convenience:

```python
# src/taxonomy/__init__.py
from .behavioral import BehavioralTaxonomy
from .structural import StructuralTaxonomy   # add this line
```

And re-export from `src/__init__.py` if you want it in the top-level namespace:

```python
from src.taxonomy.structural import StructuralTaxonomy
```
