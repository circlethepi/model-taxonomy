# Geometry Methods

A `GeometryMethod` takes a `DistanceMatrix` and embeds the models into a low-dimensional coordinate space. The resulting `GeometryResult` makes the structure of the model collection visually interpretable and enables coordinate-space queries like nearest-neighbor search.

The geometry step is optional. You can call `TaxonomyAnalyzer.fit()` with `geometry_method=None` and work with the distance matrix directly.

## Available methods

### MDSGeometry — Multidimensional Scaling

Classical or non-metric MDS via `sklearn.manifold.MDS`. This is the **recommended default** for most use cases.

```python
from src import MDSGeometry

# Metric (classical) MDS — preserves absolute distances
geo_method = MDSGeometry(
    n_components=2,
    metric=True,      # True = metric MDS, False = non-metric (ordinal)
    max_iter=300,
    random_state=0,
)
```

**Metric MDS** (`metric=True`) finds coordinates that minimize the sum of squared differences between the input distances and the Euclidean distances in the embedding. It reports a **stress** value: lower is better, with 0 meaning a perfect embedding.

**Non-metric MDS** (`metric=False`) only preserves rank order (which models are closer than which), not the actual distance values. This is more robust when the distance metric does not satisfy the triangle inequality, or when you only care about the topology of the model space.

**Stress interpretation:**
| Stress | Quality |
|---|---|
| < 0.05 | Excellent |
| 0.05–0.10 | Good |
| 0.10–0.20 | Fair — treat the embedding as approximate |
| > 0.20 | Poor — consider more components or non-metric MDS |

The stress is accessible via `result.geometry.stress`.

---

### PCAGeometry — Principal Component Analysis

Double-centers the distance matrix to produce a Gram matrix, then decomposes it with eigendecomposition. This is mathematically equivalent to **classical MDS** with Euclidean distances, but implemented via a direct eigendecomposition rather than iterative optimization — making it faster and fully deterministic.

```python
from src import PCAGeometry

geo_method = PCAGeometry(n_components=2)
```

The result includes an `explained_variance_ratio` in `geometry.metadata`:

```python
geo = pca_method.fit(distance_matrix)
print(geo.metadata["explained_variance_ratio"])
# [0.42, 0.28]  → first two components explain 70% of variance
```

**When to use PCA over MDS:**
- When speed or reproducibility matters (no random initialization).
- When you want the explained variance as a diagnostic.
- When the number of models is large (eigendecomposition is `O(N³)` but has no hyperparameter tuning).

---

### UMAPGeometry — Uniform Manifold Approximation and Projection

UMAP produces non-linear embeddings that preserve local neighborhood structure. It is more expressive than linear methods but has hyperparameters that require tuning.

```python
from src import UMAPGeometry

geo_method = UMAPGeometry(
    n_components=2,
    n_neighbors=5,     # size of local neighborhood; smaller = finer structure
    min_dist=0.1,      # minimum distance between points in the embedding
    random_state=0,
)
```

Requires the optional `umap-learn` package:

```bash
pip install umap-learn
# or
pip install "model-taxonomy[umap]"
```

**When to use UMAP:**
- When you have many models (> 50) and expect cluster structure.
- When MDS stress is high and you want a cleaner visual layout.
- When preserving global distances matters less than preserving local clusters.

**Caveats:**
- The embedding depends on `n_neighbors` and `min_dist`; results are not uniquely determined.
- UMAP distances in the embedding do not correspond to the input distances — use the distance matrix for quantitative comparisons, and UMAP only for visualization.
- With small collections (< 10 models), `n_neighbors` should be set to `N-1` or less.

---

## Comparing methods

| Property | MDS (metric) | PCA | UMAP |
|---|---|---|---|
| Preserves distances | Yes | Yes | No (local only) |
| Linear | Yes | Yes | No |
| Deterministic | No (random init) | Yes | No |
| Stress metric | Yes | No | No |
| Scales to many models | Fair | Fair | Good |
| Requires optional dep | No | No | Yes (umap-learn) |
| Recommended for | General use | Speed / reproducibility | Visual clustering |

---

## Working with geometry results

```python
result = analyzer.fit(model_ids)
geo = result.geometry

# Coordinates
print(geo.coordinates)         # np.ndarray, shape (N, n_components)
print(geo.coordinates.shape)   # (N, 2)
print(geo.stress)              # float or None (MDS only)

# Nearest neighbors in coordinate space
geo.nearest_neighbors("meta-llama/Llama-3.2-1B", k=3)
# ["meta-llama/Llama-3.2-1B-Instruct", "Qwen/Qwen2.5-1.5B", ...]

# Graph representation
import networkx as nx
import matplotlib.pyplot as plt

g = geo.to_networkx(distance_matrix=result.distance_matrix)
pos = {mid: geo.coordinates[geo.model_ids.index(mid)] for mid in geo.model_ids}
nx.draw(g, pos, labels={n: n.split("/")[-1] for n in g.nodes}, with_labels=True)
plt.show()

# Save and reload
geo.save("./results/geometry")
geo_loaded = GeometryResult.load("./results/geometry")
```

## Choosing the number of components

2 components is the standard choice for visualization. Use more components when:

- MDS stress is high with 2 components — add components until stress falls below 0.10.
- You are using the coordinates as features for downstream analysis (clustering, regression), not just visualization — in this case, retain enough components to capture most of the variance.

```python
# Diagnostic: plot stress vs number of components
stresses = []
for k in range(1, 6):
    geo = MDSGeometry(n_components=k).fit(distance_matrix)
    stresses.append((k, geo.stress))
print(stresses)
```
