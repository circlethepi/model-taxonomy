"""OLMo-2-1B instruct, the 1B rung of the suite family's scale ladder.

``allenai/OLMo-2-0425-1B-Instruct`` stands in for the checkpoint originally asked
for, ``apple/OpenELM-1_1B-Instruct``.  That repo ships no tokenizer and no chat
template at all; its attention modules are ``qkv_proj``/``out_proj`` and its FFN
``proj_1``/``proj_2``, so the repo's default target list matches zero modules and
PEFT raises; and its remote code targets transformers 4.39, calling
``DynamicCache.from_legacy_cache``, which no longer exists in the installed 5.x.

OLMo-2 at 1B is the closest match to OpenELM's 1.1B and preserves the intent -- a
small model from a family not yet represented -- while being ungated, shipping its
own tokenizer and chat template, carrying no remote code, and being natively
supported by the installed transformers.

``match`` is a complete model id rather than a family prefix, following
``LLAMA3_INSTRUCT``: ``expected_lora_params`` is a width-specific claim, and the
7B/13B OLMo-2 sizes are different widths.  A new size gets its own file.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

OLMO2 = ModelProfile(
    match="allenai/OLMo-2-0425-1B-Instruct",
    torch_dtype="bfloat16",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    prompt_format="chat",
    chat_template_sha=(
        "fe689ffbd6a4e2d0532d7480696b065b10e0e1eff3f9b9fc4bea415761e4bf4a"
    ),
    chat_template_kwargs={},  # {} == the template's own defaults
    # None deliberately, and this is a *chat* profile declining a cut -- not the
    # inert None the raw profiles carry.  This template emits <|user|> and
    # <|assistant|> as ordinary text, so there is no atomic token after the
    # question to cut at; the prompt is the template's full render, trailing
    # newline included, and ``encode_pair``'s seam assertion is what confirms
    # that boundary is safe.  It is the field's whole point that this now reads
    # as a decision rather than as an accident of the added vocabulary.
    prompt_end_token=None,
    expected_lora_params=4_194_304,
    excluded_lora_modules=(),
    # pad_token is deliberately omitted: this checkpoint declares <|pad|>
    # (100277), distinct from eos <|endoftext|> (100257), so ``apply_pad_token``
    # reports "declared by the checkpoint" and the field would be redundant.
    notes="""Prompt rendering: this template emits ``<|user|>`` and
    ``<|assistant|>`` as ordinary text rather than as registered tokens, so there
    is no atomic token after the question to cut at.  The profile declares
    ``prompt_end_token=None`` and the training prompt is the template's full
    render; see that field's comment above.

    This suite was parked from 2026-08-31 to 2026-09-02 because the code then
    derived that cut point instead of reading it.  ``render_prompt`` finished by
    cutting after the last added-vocabulary token in the rendered string, which
    here is the ``<|endoftext|>`` the template puts at index 0, so the rendered
    training prompt was the single string ``<|endoftext|>``:

        rendered:   '<|endoftext|><|user|>\\nWhy is the sky blue?\\n<|assistant|>\\n'
        after trim: '<|endoftext|>'

    Nothing downstream reported it.  Shapes stayed valid, the loss stayed finite,
    and all 16 adapters would have trained on an empty prompt while every
    behavioral probe asked the model nothing -- stage 4 checks completion-only
    loss and truncation, not that the prompt contains the question.  Smoke run
    237512 passed stages 1b-4 in exactly that state.

    The defect was in shared code and is fixed there, by declaring the cut point
    rather than deriving it.  It was never OLMo-2-specific: Qwen3.5 escaped only
    because ``<think>`` happens to be a registered token sitting after
    ``assistant\\n``, and SmolLM2-1.7B, Qwen2.5-1.5B and LFM2-1.2B were each
    measured silently losing ``assistant\\n``.  Under the declaration the three
    in-flight suites' prompts are byte-identical, so no adapter stopped being
    comparable, and this checkpoint's prompt goes from 13 characters to 65 with
    the question restored.

    Swapping to ``tiiuae/Falcon3-1B-Instruct`` was the other candidate resolution
    and is not taken.  Worth recording that it would not have worked as the
    parking notice described: Falcon3's ``<|assistant|>`` is not in its added
    vocabulary either, so declaring it as a cut point is refused by
    ``assert_compatible``, and that checkpoint would have needed ``None`` for the
    same reason this one does.

    ``allow_truncated_generation`` stays at its default False.  The parking-era
    smoke (237512) also failed stage 5 with ``termination_rate 0.00``, but that
    was measured while the model was prompted with a bare ``<|endoftext|>``, and
    a model given no question has no reason to stop.  Re-smoked on the fixed
    renderer (run 264063) the stage passes at 0.75 with median length 119 inside
    the 128-token budget, so unlike Mistral-Nemo there is no truncation here to
    declare.

    ----

    Uniform multi-head attention -- 16 query heads and 16 KV heads, no
    grouping -- so all four projections are 2048->2048 and q/k/v/o reaches all 16
    of 16 layers.  At r=16 with hidden 2048:

        16 * 16 * 4 * (2048+2048) = 4,194,304

    OLMo-2's ``q_norm`` and ``k_norm`` are RMS norms applied to the projected
    query and key, not projections.  They are correctly unmatched by the target
    list: PEFT matches by name suffix, and neither name ends in one of the four
    targets, so there is nothing to exclude and ``excluded_lora_modules`` stays
    empty.  Noted because the names read like projections at a glance.

    Capacity parity with the 4B/8B suites is BROKEN, deliberately and by a
    declared amount.  LORA_RANK = 16 is one constant for the whole experiment,
    and the qwen/llama3i suites justify their target lists by landing within ~10%
    of each other (12,386,304 and 13,631,488).  This is 4.2M -- about 0.31x the
    llama3i adapter and 0.34x the qwen one.  No ~1B model could sit in the 12-13M
    band at any sensible rank: at hidden 2048 and 16 layers, matching 12.4M needs
    r=48, which is a near-full-rank reparameterization of every projection and a
    different kind of adapter, not the same one scaled.  So this divergence is
    inherent to adding a scale ladder rather than a property of the checkpoint
    chosen, and recording it is what expected_lora_params is for.

    dtype is bfloat16, the checkpoint's native format.  As with Mistral-Nemo
    there is no sibling suite whose fp16 this has to match; 1B is a new rung.

    No reasoning block, so ``</think>`` instrumentation is inert and the
    behavioral level embeds answers directly.

    The template ships in tokenizer_config.json rather than as a standalone
    chat_template.jinja, and interpolates no date and calls no ``strftime_now``,
    so the pin is stable across days rather than merely within one.  Smoke stage
    1b re-checks it on every run.""",
)
