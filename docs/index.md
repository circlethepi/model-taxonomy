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

## Five information taxonomies

| Taxonomy | What is compared | Status |
|---|---|---|
| **Behavioral** | Mean-embedded outputs over probe inputs | Implemented |
| **Functional** | Pooled internal activations per layer, over a shared query set | Implemented |
| **Log-probability** | Per-token log-probabilities and entropies over a shared draw | Implemented |
| **Structural** | Weight matrices / LoRA adapter matrices | Implemented |
| **Dataset Embedding** | Fine-tuning dataset text distribution | Implemented |
| **Training Data** | Fine-tuning dataset distribution | Planned |

## Documentation

| Document | Contents |
|---|---|
| [Getting Started](getting_started.md) | Installation and a complete end-to-end example |
| [Core Concepts](concepts.md) | Data model, pipeline, cache hierarchy, and design rationale |
| [Behavioral Taxonomy](guides/behavioral_taxonomy.md) | Generated-output comparison, embedder strategies, probe design |
| [Functional Taxonomy](guides/functional_taxonomy.md) | Activation modes (input / generation / both), read-time views, layer selection |
| [Log-Probability Taxonomy](guides/logprob_taxonomy.md) | Teacher-forced and generation-mode log-probs, entropies, and the `07_logprobs` stage |
| [Structural Taxonomy](guides/structural_taxonomy.md) | LoRA adapter cache, config.json schema, layer selection |
| [Model Profiles and Prompt Formats](guides/model_profiles.md) | Per-checkpoint defaults, chat-template pinning, and the prompt/completion cut |
| [Experiment Suites](guides/experiment_suites.md) | `Suite`, the simplex3 config generator, and smoke-testing a new base model |
| [Cross-Level Comparison](guides/cross_level_comparison.md) | Surrogate transforms, ground-truth scoring, distributional metrics |
| [Visualization](guides/visualization.md) | Barycentric simplex colours, panel grids, regenerating the figure suite |
| [Compute Backends](guides/compute_backends.md) | Local execution and SLURM cluster setup |
| [Geometry Methods](guides/geometry_methods.md) | MDS, PCA, UMAP — when to use each |
| [Extending the Library](guides/extending.md) | Implementing a new taxonomy |
| [Dataset recipes](api_reference.md#dataset-recipes) | Recipe entry fields, `text_field` vs `text_fields`, and why the projection is part of recipe identity |
| [API Reference](api_reference.md) | Full class and method signatures |
| [Changelog](CHANGELOG.md) | History of changes |

### Working notes

`docs/notes/` holds the working notes: design records for decisions already taken,
follow-ups that were deliberately deferred, and `TODO.md`, which indexes them. It is
**gitignored** and so is not present in a clone — it is project-state and planning
material rather than documentation of the library. Anything in it that describes how the
code behaves belongs in this tree instead. `gram_and_cka.md` and `cka_notes.md` are the
two that currently sit on that line: they explain what a row means at each level, the
two CKA implementations and how they differ, and the open design question for
multi-block CKA. They are referenced from the guides but are not distributed with the
repository.
