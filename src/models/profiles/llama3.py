"""Llama 3.x base models.

These are the settings the whole ``simplex3`` suite ran under, lifted verbatim
out of ``scripts/gen_simplex3.py`` so that regenerating that suite is a no-op.
Do not "improve" them: the 16 adapters and 640 cached draws under
``meta-llama--Llama-3.1-8B`` were produced with exactly these values, and a
change here silently makes new runs incomparable to them.
"""

from __future__ import annotations

from src.models.profile import ModelProfile

LLAMA3 = ModelProfile(
    match="meta-llama/Llama-3",
    torch_dtype="float16",
    lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
    prompt_format="raw",
    # Deliberately None -- "report the count, do not assert it".  This ``match``
    # is family-wide: Llama-3.2-1B and -3B resolve here too, and they have
    # different widths, so any single number would falsely fail two of the three.
    # The 8B figure is 32 * 16 * (2*(4096+4096) + 2*(4096+1024)) = 13,631,488,
    # and it belongs on a profile whose match is a full model id -- see
    # llama3_instruct.py.  A size-specific claim needs a size-specific prefix.
    expected_lora_params=None,
    excluded_lora_modules=(),
    chat_template_sha=None,
    # None because this is a *raw* profile: there is no chat render to cut, and
    # ``render_prompt`` returns before ever reading the field.  Not to be
    # confused with OLMO2's None, which is a chat profile deliberately declining
    # a cut.
    prompt_end_token=None,
    notes="""Base (non-instruct) checkpoints: no post-training, no chat template.
    prompt_format='raw' is asserted rather than assumed -- assert_compatible
    raises if a tokenizer with a template is loaded under this profile, which is
    what stops a base model being chat-wrapped by accident.

    Note the '-Instruct' variants share this ``match`` prefix and would need
    their own profile with a longer prefix before they could be used; they are
    not covered here, and the raw assertion would catch the mistake.""",
)
