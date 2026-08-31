# Terminology

Project-specific vocabulary, in one place. These words are used throughout the
source, the guides and `docs/notes/`, and most of them were until now defined
only inline in whichever docstring happened to need them first.

This file records existing usage. It does not coin anything: every term below is
already in the code. Where a definition is subtle or has a trap attached, the
trap is stated with it, because the trap is usually the reason the word exists.

Related: [Core Concepts](concepts.md) for the pipeline these terms describe,
[API Reference](api_reference.md) for the objects that carry them.

---

## The pipeline

**Taxonomy** — *what* to extract from a model, and one of the five levels below.
A `Taxonomy` loads a model, runs inference, and returns a `ModelRepresentation`;
it unloads the model before returning, which is what keeps memory bounded when
walking a collection of large models one at a time.

**Level** — a taxonomy, used when the emphasis is on comparing across them
rather than on any one. There are five: **structural** (LoRA adapter weight
geometry, no input needed), **functional** (covariance structure of internal
activations), **log-probability** (per-token log-probs and entropies,
teacher-forced, no decoding), **behavioral** (semantic content of generated
text), and **dataset embedding** (the text distribution of the fine-tuning
mixture — no model at all). Cross-level work asks whether these five agree.

**Model representation** — the matrix `M ∈ R^{N × d}` a taxonomy extracts: `N`
probe inputs by `d` embedding dimensions.

**Probe** / **query** — the inputs a taxonomy runs a model on. The *query draw*
is the sampled set of them; see **draw**.

**Distance matrix** — the symmetric `(N, N)` matrix a `DistanceMetric` produces
over every pair in a collection. It is **self-describing**: it carries its own
`model_ids`, so the bytes are meaningful independently of the order a caller
expected. That property is what makes the row-order guard possible.

**Geometry** — the `(N, k)` coordinate array a `GeometryMethod` (MDS, PCA, UMAP)
embeds a distance matrix into. Optional; a distance matrix can be analyzed
directly. Unlike a distance matrix, a geometry is **only defined up to
rotation**, which is why geometries are refitted rather than permuted whenever
the ids do not match.

---

## The shared cache

**Shared cache** — the numbered store under `results/shared_cache`. Directories
sharing a number sit at the same stage; a **letter suffix means analysis *of*
the objects at that stage**, as in `03A_adapter_alignments`.

| Stage | Holds |
|---|---|
| `01_datasets` | Recipes and the source-row indices of each draw |
| `02_dataset_embeddings` | Dataset-level surrogates |
| `03_adapters` | Raw PEFT weights and structural representations |
| `03A_adapter_alignments` | Pairwise Procrustes alignments |
| `04_activations` | Pooled hidden states per layer |
| `05_generated` | Generated text and its embeddings |
| `05A_logprobs` | Per-token log-probabilities and entropies |
| `07_collections` | Distance matrices and geometry results |

**Draw** — one sampled subset of a dataset, spelled `n{n}_s{seed:02d}` with the
seed zero-padded. `src/cache/_draw.py` owns the spelling. Writing is narrow and
reading is wide — `draw_name` always pads, `parse_draw_name` accepts both — so
names written before the padding migration still read.

**Query draw** — the draw a model was *run on*, as opposed to the draw it was
trained on. The three inference stages (`04`, `05`, `05a`) share one key from
`DrawKeyedCache`, so one model under one query draw sits at the same coordinates
in all three and the trees can be read side by side.

**Recipe** — the specification of a dataset mixture. **`recipe_hash`** is
content-addressed: a SHA-256 of `{recipe_type, datasets}`, with the *name not
hashed*. `n` and seed live in the draw filename rather than the identity, so a
mixture swept over 19 sizes and 10 seeds is **one** recipe with ~190 draws
beside it, not 190 near-duplicate recipes. Consequence worth knowing:
`names.json` accumulates every config-block name that resolved to the hash,
because several legitimately do.

**Surrogate** — a stored derived view of a representation, living under a
`surrogates/{hash}/` subdirectory. A `02` surrogate is **authored**, not derived
from an inference run, unlike those at `04`/`05`.

**Surrogate hash** — the per-model identifier of which surrogate view a model
resolved to. A surrogate spec carries the shared *query* draw's recipe hash
rather than the model's training recipe, so in practice **every model in a
collection usually shares one surrogate hash** — a property of the data, not a
guarantee, and it breaks as soon as models are extracted against different query
datasets.

**Slice** — a sub-collection grouped by named fields, usually `(n_samples,
seed)`, from `CacheIndex.slices()`. The unit a taxonomy comparison runs over.

**Prompt format** — an optional `_f{fmt}` suffix on a draw directory, recording
which chat template was applied. Deliberately kept out of `recipe_hash`, which
would otherwise change the identity of every cached draw at once.

**View** — how a functional representation is assembled before comparison,
`concat` or `gram`. Not interchangeable; see `docs/notes/gram_and_cka.md`.

---

## Collections and their keys

**Collection** — one distance matrix plus its geometries, stored in
`07_collections` under a handle. The unit of reuse.

**Handle** — the path a collection is stored under:
`{taxonomy}/{collection_key}/{metric}_{surrogate_key}`. Composed in one place,
`collection_handle()` in `src/analysis/comparison.py`, so that
`build_taxonomy_artifacts` and the figure suite key identically.

**Collection key** — a hash over each model's `(model_id, artifact_path)` pair.
Two properties matter. It keys on **what each model actually resolved to**
rather than on the caller's selector, because `{}` and `{'draw': ...}` can be
the same read while being different dicts. And it **sorts the entries before
hashing**, which is why row order is not part of a handle — see **row-order
guard**.

**Artifact path** — a model's resolved storage path, **relative to the cache
root**, stopping before `surrogates/`. It must stay relative: an absolute path
would key a collection to one working directory. `collection_key` raises on an
absolute path rather than storing one. A useful consequence is that handles are
independent of where the cache root itself sits.

**Surrogate key** — a hash of the *ordered* list of per-model surrogate hashes.
Note the asymmetry with `collection_key`, which sorts: `surrogate_key` does
**not**. That asymmetry is currently masked because the models in a collection
share one surrogate hash, and it is recorded rather than relied upon.

**Rung** — one row of a figure grid: a resolved selector over a level, such as a
layer group or a pooling choice. A grid is **surrogate × metric**. Two surrogates of one
level read the same artifacts under the same surrogate — that is what makes them
one level — so a surrogate must reach the cache key explicitly or two surrogates collide
on a single handle.

Key on the **resolved selector dict**, never the surrogate's display label. Labels
like `"late third"` are editable prose: redefining which layers that names,
without changing the string, would leave a label-keyed entry serving a matrix
built from the old definition.

**Row-order guard** — `DistanceMatrix.reindex(model_ids)`, which permutes rows
and columns into the caller's order, selects a subset, and raises on an unknown
or repeated id. Necessary because `collection_key` sorts before hashing, so a
matrix written in one row order and one written in another land on the same
handle. The stored `model_ids` are self-describing so the bytes on disk stay
correct, but an unguarded hit hands back rows in whoever-wrote-it-first's order,
giving the caller a matrix whose labels no longer describe its rows. See
`docs/notes/row_order_bug.md`.

**Cold run** / **warm run** — a run with the collection cache disabled
(`--no-cache`, everything recomputed) versus one that reads it. "A warm run
reproduces a cold run exactly" is the only real test that the reuse is correct;
the artifact to compare is the **`matrix_sha256`** column of
`crosslevel_scores.csv`.

---

## Cross-level comparison

**Fleet** — the whole set of models under comparison, taken together. A
**fleet-level transform** is computed from all of them rather than per model.

**Surrogate transform** — a fleet-level transform applied before distancing:
`centered()` (on the fleet mean) or `whitened()` (against the fleet
covariance), from `src/analysis/surrogates.py`. These are **not alternative
metrics** — they change what is being compared. Every level carries a large
component shared by all models (the same questions, the same answer register,
the same base model) which is identical by construction and can only dilute a
similarity; the centered surrogates measure what is left.

**`transform_key`** — the short stable name a transform keys under. An anonymous
callable keys as `"custom"` and is rejected for cached collections, since two
different anonymous transforms would be indistinguishable in a handle.

**Mixture** — a model's training blend over the topic groups, e.g. 75/25. The
ground truth a taxonomy is scored against.

**Simplex** — the ground-truth geometry the mixtures live in. Its corners are
the **vertices** (the pure, single-group recipes); **barycentric weights** are a
model's coordinates within it; **anchors** are the points used to fit the
projection and **evaluation points** are those held out from it.

**Ground truth** — the mixing proportions themselves, as distinct from any
taxonomy's estimate of them.

**dCor** / **Procrustes** / **stress** — the agreement scores reported per
(surrogate, metric) cell in `crosslevel_scores.csv`. Note that `dcor_vs_truth` and
`disparity_vs_truth` run in **opposite directions**: higher is better for one,
lower for the other.

**Absent cell** — a (surrogate, metric) combination that cannot exist, drawn with its
reason in place rather than left blank or dropped, so the constraint stays
visible in the figure. The gaps are structural: CKA, MMD and energy all need
more than one row, so none can run on a `model mean` representation, and
Bures-Wasserstein stacks per-block factors before its SVD, so a selection mixing
2560-input and 4096-input projections has no BW value.
