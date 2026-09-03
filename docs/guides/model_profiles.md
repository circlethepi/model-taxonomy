# Model Profiles and Prompt Formats

Two layers describe what a *checkpoint* needs, so nothing else in the pipeline has to
know about it:

- **`src/models/profile.py`** — per-model facts: dtype, LoRA target modules, whether the
  model expects a chat template, and a hash of the template it was written against.
- **`src/datasets/_chat_projection.py`** — how a dataset row becomes a prompt and a
  completion under that template.

Both were added so that adding a second base model means *describing the model* rather
than editing a generator.

## `ModelProfile`

```python
@dataclass(frozen=True)
class ModelProfile:
    match: str                                   # an id PREFIX, not a regex
    torch_dtype: str = "float16"
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    prompt_format: str = "raw"                   # "raw" | "chat"
    chat_template_sha: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    expected_lora_params: int | None = None      # None = report, don't assert
    excluded_lora_modules: tuple[str, ...] = ()  # must NOT be matched
    pad_token: str | None = None                 # None = fall back to eos
    allow_truncated_generation: bool = False     # True = budget may be exhausted
    notes: str = ""
```

`resolve(model_id)` picks the **longest matching prefix**, so `Qwen/Qwen3.5-` wins over
a hypothetical `Qwen/` with no ordering rules to get wrong, and a `-Base` profile can be
added later without disturbing the instruct one. An unknown model is not an error: it
falls back to a bare default (fp16, q/k/v/o, raw prompts) — the settings every model in
the repo used before profiles existed.

The registry is `src/models/profiles/`: one module per family, one line in
`__init__.py`. Adding a base model should be exactly that.

### A profile is a source of defaults, never a runtime lookup

This boundary is the whole design. The generator resolves a profile **when it emits
YAML** and writes every value out explicitly; `finetune_lora.py`, `scripts/_utils.py`
and `src/taxonomy/behavioral.py` read only the YAML. So:

- an old config re-runs identically forever,
- editing a profile cannot retroactively change a past experiment,
- a run's config file stays a complete description of itself.

A profile that leaked into runtime would break all three at once.

**Three fields are exceptions, and the boundary that actually holds is narrower than the
heading.** `finetune_lora.py` resolves a profile at runtime for `pad_token`; it and
`scripts/_utils.py` both call `assert_compatible`, which reads `chat_template_sha`; and
`_chat_projection.render_prompt` reads `prompt_end_token` at render time. State the rule
as it really is:

> **A profile may be read at runtime for facts about a pinned checkpoint or template,
> never for choices that belong to an experiment.**

All three are facts of the first kind. In particular, for a template pinned by
`chat_template_sha` the correct cut point is *determined*: editing `prompt_end_token` is
a bugfix or an error, never a legitimate change to what a past experiment meant. That is
the property the rule exists to protect, and it survives.

Emitting `prompt_end_token` into the YAML would honour the heading literally, and was
rejected. It cannot go inside the `prompt_format:` block — that block feeds `format_id()`,
which is a cache key, so a new field there would rename every adapter and draw on disk.
A new top-level key would work but forces all five suites to regenerate with a real diff,
for behaviour identical to reading the profile.

Note what is deliberately *not* in a profile: the `</think>` token id. It is derived
generically from the tokenizer at runtime, returning `None` for models that have no such
token, so a model that grows a thinking mode is instrumented without a profile edit.

The same rule decides the newer fields, and it is worth stating as a rule because it is
the one that keeps this from becoming a bag of per-model constants:

> **Derive it if the checkpoint declares it. Put it in a profile only if it is a
> *choice* about the checkpoint that the checkpoint does not declare about itself.**

Derived, therefore *not* profile fields: the turn-end token ids (`stop_token_ids` in
`src/taxonomy/behavioral.py` unions pad, tokenizer eos, and every id in
`generation_config.eos_token_id`); the layer count, head count, attention layout and
`attn_output_gate` the figure suite reads straight off the model config.

Profile fields, because nothing declares them:

- **`expected_lora_params`** — what the target list *should* reach. PEFT matches by name
  suffix and errors only on *zero* matches, so a list that reaches a quarter of the depth
  is silent; the count is the check. Well defined as a bare integer only because
  `LORA_RANK = 16` in `gen_simplex3.py` is a module constant rather than a `Suite` field.
- **`excluded_lora_modules`** — modules that must stay unadapted. An exclusion nobody
  checks stops holding the moment a target list is edited.
- **`pad_token`** — which token to pad with when the checkpoint declares none. With
  pad == eos, anything masking on the pad id also masks a genuine end-of-turn, so a
  completion-only run can be taught never to end its own turn. Llama-3.x ships
  `<|finetune_right_pad_id|>` and then does not set `pad_token`, so the generic fallback
  fires on exactly the models that supply a better answer. `apply_pad_token` is the single
  definition, called from both `finetune_lora.py` and `_hf_inference.py` — resolving pad
  differently in training and extraction would mask different positions in two halves of
  one experiment. Naming an absent token, or one that *is* eos, raises.
- **`allow_truncated_generation`** — accept generations that run the whole
  `max_new_tokens` budget instead of finishing inside it. Default `False`, so the smoke
  harness's stage 5 keeps its full force everywhere it is not deliberately waived. The
  most consequential of these choices: it declares that this checkpoint's behavioral
  level embeds the *opening* of an answer where the other suites embed whole ones, which
  is a real difference from the suites it will be compared against. Only `MISTRAL_NEMO`
  sets it — it answers a Yahoo question with a numbered, headed essay and never finishes
  in 128 tokens. Accepted rather than fixed because `max_new_tokens` belongs to the
  experiment, not the `Suite`: raising it for one suite breaks the comparison that suite
  exists for, and raising it everywhere invalidates every behavioral result on disk. Set
  it only next to a `notes` entry saying why.
- **`prompt_end_token`** — the token to cut the generation prompt after, or `None` to
  keep the template's whole render. Required keyword-only, so the *unset* state does not
  exist: omitting it is a `TypeError` at import, and a profile author has to have
  decided. This replaces a derivation — `render_prompt` used to cut after whichever
  added-vocab token sat furthest right in the rendered text, on the theory that all it
  would ever remove was the whitespace after the final role marker. OLMo-2 disproved
  that: it emits `<|user|>`/`<|assistant|>` as ordinary text, so the only added-vocab
  token in its prompt is the BOS at index 0, the cut landed at character 13, and the
  training prompt was the bare string `<|endoftext|>` with the question gone. No
  derivation could get that right, because there is no atomic token after the question to
  find. `None` is the safe direction rather than the ignorant one: it keeps the whole
  render, so the question is always present, and if the untrimmed seam is not a token
  boundary `encode_pair`'s assertion raises — a careless `None` therefore fails loudly or
  is correct, and can never silently drop content. Discarding content requires naming a
  token on purpose. The named token must be atomic, which `assert_compatible` checks; see
  the section below.

  `LLAMA3` and the `resolve()` fallback spell `None` too, but mean something different by
  it — they are *raw* profiles where a cut point is not applicable, and `render_prompt`
  returns before ever reading the field. Each profile's comment says which of the two it
  means. A distinct sentinel was rejected: it reintroduces a name to invent and an unset
  state to police, for no behavioural difference.

### `assert_compatible` — template drift, checked before a GPU is held

`assert_compatible(profile, tokenizer)` raises unless the loaded tokenizer is the one the
profile describes. Two branches, because base and instruct models fail in opposite
directions:

- A **chat** profile requires a template, and requires it to hash to
  `chat_template_sha`. HuggingFace revises templates in place under the same model id
  and nothing in the repo id changes when they do — so hashing the template, rather than
  recording a version string, is what makes drift detectable. Adapters trained against
  the old template are not comparable to ones trained against the new.
- A **raw** profile requires that *no* template is in play. `meta-llama/Llama-3.1-8B` is
  a true base model with no post-training; chat-wrapping it would produce an input shape
  it has never seen, and this assertion is what stops that happening by accident.

It runs at config time rather than at first use, so it fails on any machine before a GPU
is held.

### Registered families

| Profile | `match` | dtype | Prompt format |
|---|---|---|---|
| `LLAMA3` | `meta-llama/Llama-3` | `float16` | `raw` (asserted: no template) |
| `LLAMA3_INSTRUCT` | `meta-llama/Llama-3.1-8B-Instruct` | `float16` | `chat`, template pinned |
| `QWEN3_5` | `Qwen/Qwen3.5-` | `bfloat16` | `chat`, template pinned |
| `MISTRAL_NEMO` | `mistralai/Mistral-Nemo-Instruct-2407` | `bfloat16` | `chat`, template pinned |
| `OLMO2` | `allenai/OLMo-2-0425-1B-Instruct` | `bfloat16` | `chat`, template pinned |

The first two are the longest-prefix rule doing real work rather than illustrating
itself. `meta-llama/Llama-3.1-8B-Instruct` **starts with** `meta-llama/Llama-3`, so
without the second row an instruct checkpoint resolves to a profile declaring
`prompt_format='raw'`. `assert_compatible` would catch that loudly — but only *after*
`Suite.for_model` had emitted a suite with no `prompt_format` block at all, which is a
raw suite and a different experiment. The instruct profile's `match` is a complete model
id rather than a family prefix because `expected_lora_params` is a width-specific claim
and the 3.2 sizes are a different width; a new instruct size gets its own file.

That also explains why `LLAMA3` leaves `expected_lora_params` as `None`. Its match spans
`Llama-3.1-8B`, `Llama-3.2-1B` and `Llama-3.2-3B`, so any single count would falsely fail
two models out of three. **A size-specific claim needs a size-specific prefix**; a
family-wide profile reports the count instead of asserting it.

Each carries a `notes` field with the reasoning behind its values — read it before
changing one. Two examples of what lives there:

- **Qwen3.5 is hybrid-attention**: 24 linear-attention layers and 8 full-attention
  layers in a 3:1 pattern. The repo's default `q/k/v/o` target list matches *only* the 8
  full-attention layers, and PEFT matches by name suffix and errors only on *zero*
  matches — so the default would silently adapt a quarter of the depth and record an
  identical target list in `adapter_config.json`. Adding `in_proj_qkv` / `in_proj_z` /
  `out_proj` reaches 32/32.
- **Llama-3's values are frozen deliberately**: the 16 adapters and 640 cached draws
  under `meta-llama--Llama-3.1-8B` were produced with exactly those values, and a change
  makes new runs incomparable to them.

## `PromptFormat`

`_chat_projection` is a **sibling** of `_text_projection`, not an extension of it. It is
the only place in the repo that ever calls `apply_chat_template`.

That separation is load-bearing: `row_text()` feeds `MixedDataset.to_queries` and so the
dataset-embedding level. If chat rendering entered `row_text`, the nomic centroids would
change, orphaning the cached draws in `02_dataset_embeddings` and making the dataset
level base-model-dependent, which it is not.

```python
@dataclass
class PromptFormat:
    format: str = "raw"                # "raw" | "chat"
    user_fields: tuple[str, ...] = ()
    answer_fields: tuple[str, ...] = ()
    separator: str = DEFAULT_SEPARATOR
    answer_separator: str = DEFAULT_SEPARATOR
    chat_template_kwargs: dict = field(default_factory=dict)
```

In YAML:

```yaml
prompt_format:
  format: chat
  user_fields: [question_title, question_content]
  answer_fields: [best_answer]
  chat_template_kwargs: {}     # {} == the template's own defaults
```

`chat_template_kwargs` are the *template's own* parameters — Qwen's template reads
`enable_thinking`, for instance. An empty dict means "take the template's defaults".
`from_config` requires `user_fields` when `format: chat`.

**Additive by construction.** `PromptFormat()` with no config block is `format="raw"`,
which delegates straight to `row_text`, and `to_dict()` returns `{}` — so every existing
experiment renders exactly the bytes it always did and no serialized config changes
shape.

### `format_id` and the `_f{id}` draw suffix

`format_id()` is the first 8 hex characters of a SHA-256 over `to_dict()`, or `None`
when raw. `None` is load-bearing: callers append it to draw and adapter names only when
it is not `None`, so every existing path stays exactly as it is on disk.

It exists because the inference stages can otherwise collide with themselves — same
adapter, same recipe, same `(n, seed)`, but prompts rendered through a different chat
template is a genuinely different computation, and every save in `04`/`05` is idempotent
on filename. Without the suffix, the second run silently no-ops and hands back the first
run's numbers.

It is deliberately **omitted from `01_datasets` and `02_dataset_embeddings`**, which are
keyed by recipe alone and are genuinely model-free, and deliberately **not** folded into
`recipe_hash`, which would change the identity of every cached draw at once.

### The prompt/completion cut must land on an atomic token

> The prompt/completion boundary is a token boundary at inference, therefore it must be
> a token boundary at training — and the cut belongs at an *atomic* token, so that
> tokenizing the two sides separately and tokenizing them together give the same ids.

`generate()` is handed the prompt tokenized *alone* and appends to it, so training must
show the model a prompt prefix whose ids are byte-identical to that. Anything that
tokenizes prompt and completion as one string breaks this wherever BPE merges across the
seam — and it does merge. Qwen3.5's generation prompt ends `<think>\n` and its completion
begins `\n</think>`: separately those are two `\n` (id 198); jointly they are one `\n\n`
(id 271).

```
cut after "<think>\n"   p=[...,248068,198]  c=[198,248069,...]   concat != joint
cut after "<think>"     p=[...,248068]      c=[271,248069,...]   concat == joint
```

Added tokens are never merged into their neighbours, so cutting immediately after an added
token of the generation prompt gives both properties at once. None of this is
Qwen-specific — Llama-3 instruct's generation prompt ends `<|end_header_id|>\n\n` and
takes the identical treatment.

**Which token to cut after is declared, not derived.** `ModelProfile.prompt_end_token`
names it, beside the `chat_template_sha` that pins the template the name is a fact about;
`None` means the prompt is the whole render. This module used to derive it instead — scan
the rendered text for every added-vocab token and cut after the rightmost one — and
OLMo-2 disproved the derivation. Its `<|user|>`/`<|assistant|>` are ordinary text, so the
only added-vocab token in the prompt was the BOS at index 0, the cut landed at character
13, and the training prompt was the bare string `<|endoftext|>` with the question gone;
shapes stayed valid and the loss stayed finite, so nothing reported it. Qwen3.5 survived
only by accident, because `<think>` happens to sit after `assistant\n`; templates ending
in a bare `<|im_start|>assistant\n` (SmolLM2-1.7B, Qwen2.5-1.5B, LFM2-1.2B) silently lost
`assistant\n`. No derivation could get OLMo-2 right, because there is no atomic token
after its question to find.

So `render_prompt` now only *validates* the declaration, in three places:

| Failure | Caught by | When |
|---|---|---|
| the declared token is not atomic | `assert_compatible` (added-vocab check) | config time, before a GPU is held |
| the declared token does not occur in the render | `render_prompt` | render time |
| the cut would discard non-whitespace | `render_prompt` | render time |
| the seam is not a token boundary anyway | `encode_pair`'s assertion | encode time |

The last row is why `encode_pair`'s assertion is **kept** rather than retired: declaring
the token states our *intent*, while the assertion verifies the *tokenizer* agrees. That
is a different failure mode, and the one that catches a future template revision.

The four functions. `profile` is required and keyword-only on the first three, matching
the discipline on the field itself, so that no call site can silently omit the thing that
declares the cut:

| Function | Returns | Notes |
|---|---|---|
| `render_prompt(tokenizer, row, fmt, *, profile, recipe=None)` | `str` | The single definition of "the prompt" — used by both training and extraction, so the two cannot drift apart. |
| `render_pair(..., *, profile, recipe=None)` | `(prompt, completion)` | Completion taken by **subtraction** from the full rendered conversation, never assembled from template fragments. Raises if the prompt is not a prefix. |
| `encode_pair(..., max_length, *, profile, recipe=None)` | `{input_ids, completion_mask, n_prompt_tokens, n_completion_tokens, truncated}` | **Asserts** `p_ids + c_ids == joint`. Truncation is `keep_start` and explicit, because under completion-only loss a clipped row loses *supervised* tokens. |
| `template_sha(tokenizer)` | `str \| None` | Same hash `ModelProfile.chat_template_sha` pins. |

`add_special_tokens=False` throughout: the template already emits every special token it
wants.

## See also

- [Experiment Suites](experiment_suites.md) — where a profile is resolved and written
  out, and how to smoke-test a new base model.
- [Dataset recipes](../api_reference.md#dataset-recipes) — `text_field` vs `text_fields`,
  the model-free layer below this one.
