# Log-Probability Taxonomy

The log-probability taxonomy compares models by *what they find likely*. Where the
behavioral level reads the text a model produced and the functional level reads the
hidden state it passed through, this one reads the probability the model assigned.

It is the third `HFInferenceTaxonomy` subclass, peer of `FunctionalTaxonomy` and
`BehavioralTaxonomy`, and it shares their cache key exactly — one model under one
draw sits at the same coordinates in `04_activations`, `05_generated` and
`07_logprobs`, so the three trees can be read side by side.

## Two modes, one storage stage

The level has two modes, and only one of them is a class.

| Mode | Collected by | What the numbers describe |
|---|---|---|
| `input` | `LogProbTaxonomy` | Teacher-forced pass over the shared query text |
| `generation` | `BehavioralTaxonomy(collect_logprobs=True)` | The tokens the model actually drew |

`LogProbTaxonomy` accepts `mode="input"` only, and raises otherwise. Generation-mode
log-probs are *not* a second class: the distributions they need exist only inside the
`generate()` call, so they ride along with the behavioral extraction that already
makes it. Both write into the same `07_logprobs` stage.

### Why input mode is worth having on its own

It is defined for every model over every query, needs no sampling, and compares models
on one scale nothing else in the pipeline measures. A generation-based comparison asks
*do these two models say similar things*; input mode asks *do these two models find the
same text likely*, which stays well-defined where the generations are noisy.

## What is stored

```
results/shared_cache/07_logprobs/{base_slug}/{adapter_slug}/{recipe_hash}/n{n}_s{seed}[_f{fmt}]/
    queries.json                      ← query_key + source row indices
    runs/{config_hash}.json           ← extraction provenance
    logprobs/input.safetensors
    logprobs/{variant_token}.safetensors
```

Generation filenames reuse `GeneratedTextCache.variant_token` — the *same function
object*, not a reimplementation — so a log-prob file carries the same token as the
`generations/{token}.json` it describes and the two join by name with no lookup. Every
result-changing axis (token budget, replicate count, sampling settings) is in that
name, which matters because `save_logprobs` is idempotent on filename: an axis left out
of the name would make a second run at a different setting a silent no-op returning the
first run's numbers.

**Rows are padded and query-major**, matching the behavioral matrix layout exactly, so
row *i* means the same thing in both stages. `lengths` gives the real extent of each
row; positions at or beyond it are padding and carry no meaning. `content_start` (input
mode) is the first non-scaffolding position under the chat template. The index is stored
rather than the rows pre-trimmed, so the cache stays a superset: the trimmed view is
recoverable from the full rows, not the other way round.

### Two distributions in generation mode, not one

| Array | Comes from | Comparable across a temperature sweep? |
|---|---|---|
| `logprob` / `entropy` | the *processed* logits — temperature- and top-p-warped, i.e. the distribution the token was actually drawn from | no |
| `logprob_raw` / `entropy_raw` | the unprocessed model output | yes |

The warped pair is not recoverable from the raw pair, because
`log softmax(z/T)[i] = z[i]/T − logsumexp(z/T)` needs the whole ~248k-token logit
vector, which is discarded. Both are therefore stored.

Input mode is teacher-forced with no decoding at all, so it has no processed/unprocessed
split: its `logprob` *is* the raw quantity, on the same scale as `logprob_raw`.

## Configuration

```python
from src.taxonomy import LogProbTaxonomy
from src.cache.logprob_cache import LogProbCache

taxonomy = LogProbTaxonomy(
    queries=probes,
    query_key={"recipe_hash": ..., "n_samples": 1000, "seed": 0},
    cache=LogProbCache("results/shared_cache"),
    device="cuda",
    batch_size=8,          # rows per forward pass — memory only, not in the key
    mode="input",          # the only accepted value
    max_length=512,        # IS in the key: truncating elsewhere scores a different span
    seq_chunk=64,          # log_softmax window — memory only, not in the key
    torch_dtype=torch.float16,
)
```

Both `cache` and `query_key` are **required**. The per-token arrays are the stored
artifact and the returned representation is a summary of them, so there is no
in-memory-only path.

`config_dict()` carries `{taxonomy, query_key, n_queries, mode, max_length,
torch_dtype}`. `taxonomy` is in it so a log-prob config can never hash equal to a
functional config over the same draw; `seq_chunk` and `batch_size` are out because they
cannot change the numbers.

## Memory, not time, is the cost

Per-token log-probs need logits at *every* position, and a modern vocabulary is ~250k
wide: at batch 16 × seq 512 that tensor is ~4 GB in bf16, and a float32 `log_softmax`
of it ~8 GB more. The softmax is therefore chunked over the sequence axis and the
realized token gathered per chunk (`_score_chunked`) — identical numbers, bounded
memory. `seq_chunk` is the knob; lower it before lowering `batch_size`.

## What `extract` returns

An `(n_queries, 2)` `ModelRepresentation` whose columns are the per-query **mean
log-probability** and **mean entropy** over the content positions. That is the smallest
summary that makes the level usable by the ordinary representation machinery, and it is
recoverable from the stored rows rather than the other way round.

For the full detail, read the cache directly:

```python
cache = LogProbCache("results/shared_cache")
arrays, meta = cache.load_logprobs(base_id, adapter_id, query_key, "input")
per_query = LogProbCache.masked_mean(
    arrays["logprob"], arrays["lengths"], start=arrays["content_start"]
)
```

`masked_mean` is the one reduction every reader of this stage needs, kept in the cache
class so the padding convention is applied in one place rather than re-derived at each
call site. `list_entries(base, adapter, query_key)` enumerates the stems stored for one
draw (`input`, `generation128_8r_…`) as a directory listing with no file opens — the
peer of `GeneratedTextCache.list_variants`.

## See also

- [Behavioral Taxonomy](behavioral_taxonomy.md) — `collect_logprobs`, and the sampling
  settings that appear in a generation-mode filename.
- [Functional Taxonomy](functional_taxonomy.md) — the peer level that reads the hidden
  state from the same forward pass.
- [Core Concepts](../concepts.md#the-shared-cache-layout) — the shared cache layout.
