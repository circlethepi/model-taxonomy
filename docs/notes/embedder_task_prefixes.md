# nomic task prefixes, and what the pre-fix embeddings are

**Status:** fixed as of the commit that added this note. The cache situation below is
permanent and deliberate — it is not a bug waiting to be tidied up.

## What was wrong

`nomic-embed-text-v1.5` is trained with a mandatory task-instruction prefix
(`search_document: `, `search_query: `, `clustering: `, `classification: `).
This repo asked for one and never got it.

The mechanism is worth understanding, because nothing anywhere reported a problem.
`nomic-embed-text-v1.5`'s `config_sentence_transformers.json` contains **only** a
`__version__` block — there is no `prompts` map. sentence-transformers then
synthesises one, `{"query": "", "document": ""}`, whose values are the **empty
string**. So `encode(..., prompt_name="document")` looked up a key that existed, got
`""`, and prepended nothing. No exception, no warning, and output vectors that were
entirely plausible.

Every dataset embedding written before the fix — 520 entries, all nomic, all
`prompt_name: document`, all `representation: mean`, n from 1 to 140 000 — was
therefore computed on bare text.

## What that did, and did not, do to the results

Measured directly: the same 100 rows of each of the five yahoo mixtures (seed 0),
embedded both ways.

| | bare (what is cached) | `search_document: ` |
|---|---|---|
| MDS axis vs true mixing, pearson | 0.9977 | 0.9976 |
| spearman | 1.0000 | 1.0000 |
| off-diagonal distance spread | 0.0074–0.0773 | 0.0044–0.0484 |

- The **vectors** move materially: centroid cosine 0.94–0.96 between the two.
- Every pairwise distance **shrinks by roughly a constant factor**, because the prefix
  contributes a component shared by every text, which inflates all similarities.
- The **geometry does not change**: distance-matrix agreement is pearson 0.9998, and
  recovery of the mixing proportion is identical to four decimals.

MDS and any rank-based statistic are invariant to that common component. So the
cached embeddings are internally consistent and the results derived from them stand.
This was a real misuse of the model, and it is worth fixing going forward — but it is
not a retraction.

*(The comparison above ran with `max_seq_length` capped at 512, which truncates ~3% of
yahoo answers — identically in both arms, so the like-for-like conclusion holds. See
the caps table below.)*

## Why the old entries are unreachable, and why that is correct

`SentenceTransformerEmbedder.config_dict()` now includes **`prompt_prefix`**, the
resolved literal, alongside `prompt_name`. Keying on the name alone cannot tell a
bare-text embedding from a correctly-prefixed one — same `model_name`, same
`prompt_name`, different vectors — so they would share an `embedder_hash` and the
cache would hand back one where the other was asked for.

The consequence is that every pre-fix entry now has a different key and will not be
found. **That is the point.** Their distances live on a different scale, so mixing
them with prefixed ones in a single comparison is silently wrong. A cache miss here
is the system working.

- The old entries are **left on disk** and remain readable and reproducible.
- Re-embed a slice **only when something actually needs it**. A full re-embed is
  ~2.0M texts — feasible now as a single GPU job, since the embedder is no longer
  pinned to CPU — but deliberately not being done.
- The cache is untracked and local to this machine, so this note exists for us.

## If you batch a re-embed

`SentenceTransformerEmbedder.embed()` encodes **one text per call**, so the batch
dimension is 1 and `max_seq_length` costs nothing however large it is. Nothing in the
pipeline is at risk today.

It becomes a problem the moment anyone batches — which is the obvious way to make a
re-embed practical. sentence-transformers pads a batch to its longest member, bounded
by `max_seq_length`, and nomic's default is **8192**. Encoding 32 long yahoo answers
at that bound was killed by the OOM killer on a login node.

Measured over the 1000-row `050t0_050t1` draw (mean ~97 tokens, p99 ~727, max ~975):

| `max_seq_length` | yahoo answers truncated | batched memory ceiling |
|---|---|---|
| 8192 (nomic default) | 0% | very large |
| 2048 | 0% | 4× smaller |
| **1024 — recommended** | **0%** | 8× smaller |
| 512 | ~2.9% | 16× smaller |
| 128 | ~23% | — |

**Cap at 1024.** It truncates nothing while cutting the memory ceiling 8×. 512 is
tempting and silently drops the tail of about 3% of answers, changing their
embeddings — the same category of quiet wrongness this note is about.

## Behaviour now

- `prompt_name` omitted → defaults to `search_document`, with a `warnings.warn`
  naming the prefix chosen. Corpus text is what this repo embeds at both the dataset
  and behavioral levels, so it is the right default for both. Defaulting rather than
  raising is deliberate: the bug was *no prefix at all*, a misuse of the model;
  choosing `search_document` over `search_query` is a far smaller distinction, since
  all of them use the model as trained.
- An **unknown** `prompt_name` raises. A typo must not quietly take the default.
- Models not in `_PREFIX_REQUIRED_MODELS` are untouched: `prompt_prefix` is `""` and
  `prompt_name` still passes through to `encode` as before.

Guarded by `t_embedder_prefix_resolved` and `t_embedder_prefix_in_cache_key` in
`scripts/check_analysis.py`. Both assert on the *resolved literal*, not on the name —
asserting the name is exactly what would have passed all the way through the bug.

## Open question

Whether the behavioral level should use `clustering: ` rather than
`search_document: `. Grouping model outputs by similarity is arguably what
`clustering` is for. `search_document` is what keeps behavioral consistent with the
dataset level, and cross-level comparability is the whole reason to match embedder
configs. Changing it later costs a re-extract, not a redesign.
