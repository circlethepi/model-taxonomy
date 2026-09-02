"""Mistral-Nemo-12B instruct, the 12B rung of the suite family's scale ladder.

``mistralai/Mistral-Nemo-Instruct-2407`` stands in for the checkpoint originally
asked for, ``mistralai/Ministral-3-14B-Instruct-2512``.  That repo is fp8-quantized
and a *multimodal* ``Mistral3ForConditionalGeneration``, which
``AutoModelForCausalLM`` has no mapping for -- so all three of the repo's load
sites would fail -- and whose Pixtral vision tower carries ``q_proj``/``k_proj``/
``v_proj``/``o_proj`` in 24 further layers, which the default target list would
silently adapt alongside the text tower.

Nemo has **identical text-tower dimensions**: 40 layers, hidden 5120, 32 query
heads, 8 KV heads, head_dim 128.  The adapter is parameter-for-parameter the one
the 14B would have produced, while the checkpoint is bf16, ungated, free of
remote code, and loadable on the existing path with no code change.  The
architecture is preserved; only the multimodal wrapper and the fp8 packaging are
dropped.

``match`` is a complete model id rather than a family prefix, following
``LLAMA3_INSTRUCT``: ``expected_lora_params`` is a width-specific claim, and the
other Mistral widths are different widths.  A new size gets its own file.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

MISTRAL_NEMO = ModelProfile(
    match="mistralai/Mistral-Nemo-Instruct-2407",
    torch_dtype="bfloat16",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    prompt_format="chat",
    chat_template_sha=(
        "e4676cb56dffea7782fd3e2b577cfaf1e123537e6ef49b3ec7caa6c095c62272"
    ),
    chat_template_kwargs={},  # {} == the template's own defaults
    # The generation prompt ends exactly at "[/INST]", which is atomic, so
    # nothing is discarded at all.
    prompt_end_token="[/INST]",
    expected_lora_params=19_660_800,
    excluded_lora_modules=(),
    pad_token="<pad>",
    allow_truncated_generation=True,
    notes="""Uniform grouped-query attention, so q/k/v/o reaches all 40 of 40
    layers -- no hybrid layout to work around as in Qwen3.5, and no non-standard
    projection names as in OpenELM.  At r=16 with hidden 5120, 32 query heads at
    head_dim 128 (q out 4096) and 8 KV heads (k/v out 1024):

        40 * 16 * [(5120+4096) + 2*(5120+1024) + (4096+5120)] = 19,660,800

    Capacity parity with the 4B/8B suites is BROKEN, deliberately and by a
    declared amount.  LORA_RANK = 16 is one constant for the whole experiment,
    and the qwen/llama3i suites justify their target lists by landing within
    ~10% of each other (12,386,304 and 13,631,488).  This is 19.7M -- about 1.44x
    the llama3i adapter and 1.59x the qwen one.  Recorded here rather than tuned
    away because that is what expected_lora_params exists for: capacity becomes a
    confound the reader can see, not one hidden behind a per-suite rank.  The
    divergence is inherent to adding a scale ladder, not to this checkpoint.

    dtype is bfloat16, the checkpoint's native format.  Unlike LLAMA3_INSTRUCT
    there is no sibling suite whose fp16 this has to match -- 12B is a new rung
    and nothing on disk is prompting a deviation from upstream.

    pad_token: the tokenizer declares <pad> at id 10 and then does NOT set
    ``pad_token``, so the generic fallback would give pad == eos == </s>.  That
    is exactly the arrangement LLAMA3_INSTRUCT added this field to avoid --
    masking on the pad id would also mask the genuine end of turn -- and it would
    differ from qwen and llama3i, adding a second variable to a comparison meant
    to have one.

    No reasoning block, so ``</think>`` instrumentation is inert and the
    behavioral level embeds generations directly.

    allow_truncated_generation is True, and it is the sharpest caveat on this
    suite.  Smoke stage 5 measured termination_rate 0.00: every generation ran
    the full 128-token budget, cut mid-sentence.  This checkpoint answers a Yahoo
    question with a numbered, headed essay -- "The sky appears blue due to ...
    Here's a simple explanation: 1. **Sunlight**: ..." -- and 128 tokens does not
    reach the end of one.  llama3i and qwen finish inside the same budget, so
    this suite's behavioral level embeds the *opening* of an answer where theirs
    embed whole ones.

    Accepted rather than fixed, deliberately.  max_new_tokens is a property of
    the experiment and not a Suite field: raising it for this suite alone would
    make the behavioral level incomparable to the three suites this one exists to
    sit beside, and raising it everywhere would invalidate every behavioral
    result already on disk.  So the truncation is declared here and read at every
    run, the same treatment the capacity divergence above gets, for the same
    reason -- a confound the reader can see beats one that is hidden.

    What this does NOT excuse: the structural and functional levels are
    unaffected (they read weights and activations, not decoded text), and greedy
    and the temperature sweep are affected exactly as the behavioral level is.
    Read cross-suite behavioral comparisons involving this suite with that in
    mind, or restrict them to the levels that do not decode.

    The template ships in tokenizer_config.json rather than as a standalone
    chat_template.jinja.  Checked rather than assumed, because that is the shape
    that breaks a template pin: it interpolates no date and calls no
    ``strftime_now``, so the rendering is stable across days and not merely
    within one.  Smoke stage 1b re-checks it on every run.

    Prefetch trap, handled in the suite rather than here: this repo publishes its
    weights twice -- five sharded model-0000N-of-00005.safetensors AND a
    consolidated.safetensors -- so the prefetch job's '*.safetensors' pattern
    fetches ~24.5 GB more than it needs.  The nemo suite sets
    ``prefetch_ignore=("consolidated.safetensors",)``.""",
)
