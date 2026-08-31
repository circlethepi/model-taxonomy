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

Added tokens are never merged into their neighbours, so cutting immediately after the
last added token of the generation prompt gives both properties at once. None of this is
Qwen-specific — Llama-3 instruct's generation prompt ends `<|end_header_id|>\n\n` and
takes the identical treatment.

The four functions:

| Function | Returns | Notes |
|---|---|---|
| `render_prompt(tokenizer, row, fmt, recipe=None)` | `str` | The single definition of "the prompt" — used by both training and extraction, so the two cannot drift apart. |
| `render_pair(...)` | `(prompt, completion)` | Completion taken by **subtraction** from the full rendered conversation, never assembled from template fragments. Raises if the prompt is not a prefix. |
| `encode_pair(..., max_length)` | `{input_ids, completion_mask, n_prompt_tokens, n_completion_tokens, truncated}` | **Asserts** `p_ids + c_ids == joint`. Truncation is `keep_start` and explicit, because under completion-only loss a clipped row loses *supervised* tokens. |
| `template_sha(tokenizer)` | `str \| None` | Same hash `ModelProfile.chat_template_sha` pins. |

`add_special_tokens=False` throughout: the template already emits every special token it
wants.

## See also

- [Experiment Suites](experiment_suites.md) — where a profile is resolved and written
  out, and how to smoke-test a new base model.
- [Dataset recipes](../api_reference.md#dataset-recipes) — `text_field` vs `text_fields`,
  the model-free layer below this one.
