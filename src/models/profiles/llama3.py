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
    chat_template_sha=None,
    notes="""Base (non-instruct) checkpoints: no post-training, no chat template.
    prompt_format='raw' is asserted rather than assumed -- assert_compatible
    raises if a tokenizer with a template is loaded under this profile, which is
    what stops a base model being chat-wrapped by accident.

    Note the '-Instruct' variants share this ``match`` prefix and would need
    their own profile with a longer prefix before they could be used; they are
    not covered here, and the raw assertion would catch the mistake.""",
)
