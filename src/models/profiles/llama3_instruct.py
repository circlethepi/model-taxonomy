"""Llama 3.x *instruct* checkpoints, prompted through their own chat template.

Separate from :mod:`src.models.profiles.llama3` because of the prefix, not
because of taste.  ``LLAMA3`` matches ``meta-llama/Llama-3`` and declares
``prompt_format='raw'`` with an *asserted* absence of any chat template --
``meta-llama/Llama-3.1-8B-Instruct`` starts with that prefix.  Without this
file, ``resolve`` hands an instruct checkpoint the base model's profile:
``assert_compatible`` would catch it loudly, but only after ``Suite.for_model``
had already emitted a suite with no ``prompt_format`` block at all, which is a
raw suite and not the experiment.

``resolve`` picks the **longest** matching prefix, so a full model id here beats
the family prefix there.  This is the case ``llama3.py`` anticipates in so many
words: "the '-Instruct' variants share this ``match`` prefix and would need
their own profile with a longer prefix before they could be used".

The match is a complete model id rather than a family prefix, deliberately.
``expected_lora_params`` is a width-specific claim, and the 3.2 sizes are a
different width; a prefix broad enough to catch them would make that number
wrong for two models out of three.  A new instruct size gets its own file.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

LLAMA3_INSTRUCT = ModelProfile(
    match="meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype="float16",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    prompt_format="chat",
    chat_template_sha=(
        "e10ca381b1ccc5cf9db52e371f3b6651576caee0a630b452e2816b2d404d4b65"
    ),
    chat_template_kwargs={},  # {} == the template's own defaults
    expected_lora_params=13_631_488,
    excluded_lora_modules=(),
    pad_token="<|finetune_right_pad_id|>",
    notes="""Instruct sibling of meta-llama/Llama-3.1-8B, which is the checkpoint
    the existing raw ``simplex3`` suite trained on.  That pairing is the point:
    against ``simplex3`` this is a one-variable change (base -> instruct, raw ->
    chat, same weights lineage), and against ``simplex3_qwen`` it is a
    one-variable change (model family, same prompting regime).  Neither
    comparison exists if this is prompted raw, which is what the longest-prefix
    relationship to LLAMA3 protects.

    Uniform grouped-query attention, so unlike Qwen3.5 there is no hybrid layout
    and q/k/v/o reaches all 32 of 32 layers.  At r=16 with hidden 4096 and 8 KV
    heads (k/v out 1024):

        32 * 16 * (2*(4096+4096) + 2*(4096+1024)) = 13,631,488

    within ~10% of Qwen3.5-4B's 12,386,304, which is what makes the cross-suite
    capacity comparison honest.  Qwen's in_proj_* targets are linear-attention
    modules that do not exist here and would error at PEFT match time.

    float16 rather than the checkpoint's native bfloat16: this suite sits beside
    the meta-llama--Llama-3.1-8B run, and comparability with it is worth more
    than matching the upstream dtype.  Qwen3.5 deviates to bfloat16 for a
    specific architectural reason -- a recurrent linear-attention state whose
    range fp16 handles badly -- and no such state exists in a uniform-attention
    Llama.

    pad_token: this checkpoint ships <|finetune_right_pad_id|> (128004) and then
    does not set ``pad_token``, so the pipeline's generic fallback would train it
    with pad == eos == <|eot_id|>.  That is the arrangement finetune_lora warns
    about -- masking on the pad id would also mask the genuine end-of-turn, i.e.
    teach a completion-only run never to end its turn -- and it would differ from
    Qwen3.5, which has a distinct pad, adding a second variable to a comparison
    meant to have one.

    Turn end vs sequence end: ``tokenizer.eos_token_id`` is <|eot_id|> (128009),
    but ``generation_config.eos_token_id`` is [128001, 128008, 128009] --
    <|end_of_text|>, <|eom_id|>, <|eot_id|>.  ``generate`` stops on all three, so
    the log-prob trim has to as well; that is ``stop_token_ids`` in
    ``src/taxonomy/behavioral.py``, derived from the checkpoint rather than
    recorded here.

    No reasoning block, so ``</think>`` instrumentation is inert and the
    behavioral level embeds answers directly.

    The template pin is what keeps adapters comparable across reruns: HuggingFace
    revises templates in place under an unchanged model id.  This one ships in
    tokenizer_config.json rather than as a standalone chat_template.jinja, and it
    interpolates a "Cutting Knowledge Date"/"Today Date" header.  Checked rather
    than assumed, because that is exactly the shape that breaks a template pin:
    the template contains no ``strftime_now``, and ``date_string`` defaults to the
    literal "26 Jul 2024", so the rendering is stable across days and not merely
    within one.  Smoke stage 1b re-checks it on every run.""",
)
