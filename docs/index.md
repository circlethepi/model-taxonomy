# model-taxonomy

A Python library for analyzing collections of machine learning models through the lens of geometric similarity. Models are compared at one or more **information taxonomies** — levels of abstraction at which their representations can be extracted and measured.

## What it does

Given a collection of HuggingFace language models and a set of probe inputs, `model-taxonomy` computes a pairwise **distance matrix** between the models and embeds them into a low-dimensional **coordinate space**. This lets you answer questions like:

- Which models produce the most similar outputs to each other?
- Does fine-tuning a model move it closer to or farther from related models?
- How does model similarity at the behavioral level compare to similarity at the weight level?

## The three-step pipeline

Every analysis follows the same three-step structure, each step independently configurable:

```
Collection of models
        │
        ▼  Step 1: Taxonomy (Surrogate extraction)
        │  For each model, run inference over probe inputs
        │  and extract a matrix representation M ∈ R^{N × d}
        │
        ▼  Step 2: DistanceMetric (Pairwise distances)
        │  Compute a scalar distance between every pair of
        │  matrix representations → NxN distance matrix
        │
        ▼  Step 3: GeometryMethod (Coordinate embedding)
           Embed the distance matrix into a low-dimensional
           coordinate space → (N, k) coordinates
```

## Four information taxonomies

| Taxonomy | What is compared | Status |
|---|---|---|
| **Behavioral** | Mean-embedded outputs over probe inputs | Implemented |
| **Functional** | Gram matrices of internal activations per layer | Implemented |
| **Structural** | Weight matrices / LoRA adapter matrices | Implemented |
| **Training Data** | Fine-tuning dataset distribution | Planned |

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](getting_started.md) | Installation and a complete end-to-end example |
| [Core Concepts](concepts.md) | Data model, pipeline, cache hierarchy, and design rationale |
| [Behavioral Taxonomy](guides/behavioral_taxonomy.md) | Generated-output comparison, embedder strategies, probe design |
| [Functional Taxonomy](guides/functional_taxonomy.md) | Activation modes (input / generation / both), Gram matrices, layer selection |
| [Structural Taxonomy](guides/structural_taxonomy.md) | LoRA adapter cache, config.json schema, n_components |
| [Compute Backends](guides/compute_backends.md) | Local execution and SLURM cluster setup |
| [Geometry Methods](guides/geometry_methods.md) | MDS, PCA, UMAP — when to use each |
| [Extending the Library](guides/extending.md) | Implementing a new taxonomy |
| [API Reference](api_reference.md) | Full class and method signatures |
