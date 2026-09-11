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

## The simplex experiments

**`DataSimplexSpec`** — the dataset-side counterpart of
`src/experiments/suite.py::Suite`, defined in
`src/experiments/data_simplex_spec.py`. Where a `Suite` says *how* a run is
configured — base model, dtype, LoRA targets, walls, sharding — a
`DataSimplexSpec` says *what corpus the simplex is built over*: dataset id,
vertex axis, group partition, field projections, grid denominator, draw sizes,
embedder, and the prose caveats emitted into that dataset's configs.
`scripts/gen_simplex3.py` is a function of the pair, which is what lets a second
corpus be a spec entry rather than a forked generator.

**Vertex axis** — the categorical column whose values become the pure corners of
the simplex. Yahoo's is `topic`, dolly's is `category`, oasst1's is `lang`. It is
the `class_field` of every emitted recipe; the separate name exists because
"class field" says where it lives rather than what it does.

**Grid denominator** — the integer `G` such that mixtures are drawn at every
multiple of `1/G`. All three datasets use 4, i.e. the 25% grid. The grid points
are the compositions of `G` into `K` parts: 15 at `K=3`, 35 at `K=4`. The even
mixture is appended as an extra point only when `K` does not divide `G` — which
is why yahoo has 16 proportions and not 15, and why dolly and oasst1 have exactly
35.

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
| `06_pairwise` | Individual pairwise distances, one entry per model pair |
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

**nsweep** — a sweep over `n_samples`, the size of the *training* draw, holding
the mixture grid, the seeds and every optimizer setting fixed. The word predates
the simplex work: `results/yahoo_topics_mean_cosine_10seeds_nsweep` and
`experiments/archive/yahoo_topics_mean_cosine_{3,10}seeds_nsweep.yaml` are a
two-vertex, dataset-level-only version of the same axis. The trap is that a
training nsweep is not the same object as the *embedding* sweep already spelled
`n_samples_sweep:` in every emitted build config — that one varies the draw a
dataset representation is computed over, and exists at every `n` regardless of
what was trained. See `docs/notes/nsweep_dataset_size.md`.

**rsweep** — a sweep over the LoRA **rank**, holding the corpus, the mixture
grid, the training draw, the seeds and every other optimizer setting fixed. Named
on 2026-09-11 for its parallel with [nsweep](#nsweep) above: `n` there is the
training draw size, `r` here is the rank. It reaches code as `Suite.lora_ranks`,
a Suite field rather than a `DataSimplexSpec` one, because rank says how a run is
configured and not what corpus it is run over — the same split that puts
`target_modules` and `torch_dtype` on the Suite.

Two things the word does not mean. It is not a sweep over `lora_alpha`: alpha is
the *gain* on the low-rank update (PEFT scales `B @ A` by `alpha / rank`), and
the generator renders `2 * rank`, which is the 32 every adapter trained at rank
16 already carries — so an rsweep holds the gain at 2 and varies capacity alone.
And it is not a re-run of the fixed-rank tree at its own middle point: rank is
inside every adapter directory name (`..._r16_i00_b5008_f…`) and every cache is
keyed on that name, so the `r16` point of an rsweep resolves to the adapters and
activations `simplex3_olmo2` already wrote. See
`docs/notes/rank_sweep.md`.

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

### The settled triple

These three were ambiguous and are now fixed. They are the spine of the caching
design, because a `06_pairwise` handle addresses exactly one **perspective**.

| term | meaning |
|---|---|
| **selector** | The collection of hyperparameters that produce a surrogate — mode, pooling, layers, projections, view, normalize, replicate reduction, the query draw, the embedder, and any fleet transform. |
| **surrogate** | What a selector produces: the representation view being compared. One **row** of the figure grid, e.g. `late third · centered`. |
| **perspective** | A surrogate together with a similarity metric. One **cell** of the figure grid, e.g. `late third · centered` × `cosine`. |

**Canonical perspective** — a perspective designated as a *default reading* of a
taxonomy level, so that levels can be compared one-to-one without sweeping
metrics. There are seven over five levels: one each at the dataset and
behavioral levels, three **scopes** at the structural level and two at the
functional one.

| perspective | level | surrogate | metric |
|---|---|---|---|
| `dataset_embedding` | dataset embedding | mean embedding (`representation="mean"`, n=1000, seed 0) | euclidean |
| `structural_all_o` | structural | every layer, output projections only | cosine |
| `structural_all_qkvo` | structural | every layer, all four LoRA projections | cosine |
| `structural_last_o` | structural | the last decoder layer's output projection | cosine |
| `functional_all` | functional | every hidden state, input mode, mean pooling | cosine |
| `functional_last` | functional | the final hidden state (`h{N_LAYERS}`) alone | cosine |
| `behavioral` | behavioral | per-query replicate mean at R=16 | cosine |

A level carries more than one row when it can be read at more than one **scope**
and the scopes answer different questions — structural narrows by layer *and* by
projection, functional only by layer, and neither has a reason to move together
as an experiment's other axes change. The functional level has no projection
axis at all: it reads hidden states, so its mirror of the structural
all-versus-last contrast is exactly its two rows.

They are fixed in `canonical_perspectives()` in
`figures/simplex_collection_size/sweep_group_size.py` and are the standing
default for new experiments: compute these seven, score each against the
ground-truth simplex with dcorr and Procrustes disparity, and add an eighth only
after asking. The dataset level's normalised sibling metric (`frobenius`, which
L2-normalises each mean before subtracting) is a second legitimate reading of
"euclidean norm" and the simplex3 drivers carry both; whether a given suite
wants it is a per-suite decision, not part of this list.

Two things this term is **not**. It is unrelated to `_canonical(dm)` in
`src/analysis/comparison.py`, which means a distance matrix's *sorted row order*
— a different layer of the problem entirely, which is why this is always written
as the two-word phrase and never as bare "canonical". And it is not a claim that
the chosen perspective is the *best* one at its level; it is the one the project
agreed to read by default, chosen so that a level's number means the same thing
across runs.

**nsamples_train** — one value of `n_samples`, i.e. one size of the *training*
draw, as varied by an [nsweep](#nsweep). Named on 2026-09-10, replacing the
"rung" this axis had been called informally in `docs/notes/nsweep_dataset_size.md`
since 2026-09-07. The rename is because `rung` was already carrying two unrelated
senses in this project — see the note directly below — and a third would have
made the word useless. It reaches code as the `train_sizes` / `extract_sizes`
tuples on `DataSimplexSpec`, not as a field of its own; the term names the axis,
not a variable.

**`rung` is retired**, replaced by `surrogate`. This was a unification rather
than a rename: `surrogate` already meant "a read-time view of a stored artifact,
computed on demand and written back", the module applying fleet transforms was
already `src/analysis/surrogates.py`, and the figure suite's own docstring
section was already headed "Surrogate rungs". The two words were describing one
thing at two layers.

One live use survives the retirement and is *not* covered by it: the **scale
ladder** of base model sizes, where `src/models/profiles/olmo2.py` calls itself
"the 1B rung" and `mistral_nemo.py` "the 12B rung" of a 1B/4B/8B/12B family. That
is a third sense again — neither a way of reading a level nor a training draw
size — and it is left alone because it is confined to the model profiles and
`docs/guides/experiment_suites.md`, where no other ladder is in scope. If it ever
leaves those files, rename it there rather than reviving `rung` generally.

One caveat to keep in mind: the cached `surrogates/{hash}/` in `04`/`05` covers
view, pooling and normalize, while the *fleet* transform is applied later in
`src/analysis/surrogates.py`. Both are surrogates, produced at different stages.

Two surrogates of one level read the same artifacts under the same *stage*
surrogate — that is what makes them one level — so a surrogate must reach the
cache key explicitly, or two of them collide on a single handle.

Key on the **resolved selector dict**, never the surrogate's display label.
Labels like `"late third"` are editable prose: redefining which layers that
names, without changing the string, would leave a label-keyed entry serving a
matrix built from the old definition.

**Pair id** (`pair_id`) — the readable key of one entry in a `pairs.json`,
naming an unordered pair of models: `"__".join(sorted([model_id_a,
model_id_b]))`. It is an `_id` and not a `_key` because this codebase draws that
line: `model_id`, `recipe_id` and `adapter_name` are readable, while
`collection_key`, `surrogate_key` and `recipe_hash` are opaque digests.

**Selector key** — the digest of a selector, named to match `collection_key` and
`surrogate_key`.

**Selector slug** — the readable, **non-identifying** prefix on a `06_pairwise`
handle's surrogate component, e.g. `input_mean_concat_layer_L0-28`. Identity
lives entirely in the selector key, which is what lets the slug be improved later
without orphaning a single stored pair.

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
