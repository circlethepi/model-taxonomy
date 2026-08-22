"""Per-model facts, in one place per model family.

Three things vary by base model and nothing else in the pipeline should have to
know about them: what dtype the checkpoint wants, which module names LoRA should
attach to, and whether the model expects its input wrapped in a chat template.
Before this module those lived as literals inside ``scripts/gen_simplex3.py``,
which meant adding a second base model meant editing the generator rather than
describing the model.

**A profile is a source of defaults for spec construction, plus the template
assertion below.  It is never consulted to decide what a stored run meant.**

That boundary is the whole design.  The generator resolves a profile when it
emits YAML and writes every value out explicitly; ``finetune_lora.py``,
``scripts/_utils.py`` and ``src/taxonomy/behavioral.py`` read only the YAML.  So
an old config re-runs identically forever, editing a profile cannot retroactively
change a past experiment, and a run's config file stays a complete description of
itself.  A profile that leaked into runtime would break all three at once.

Note what is deliberately *not* here: the ``</think>`` token id.  It is derived
generically from the tokenizer at runtime (``convert_tokens_to_ids``), returning
``None`` for models that have no such token, so a model that grows a thinking
mode does not need a profile edit to be instrumented.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """What one model family needs that the pipeline cannot infer.

    ``match`` is an id prefix, not a regex: the registry picks the longest
    matching prefix, so ``Qwen/Qwen3.5-`` wins over a hypothetical ``Qwen/``
    without any ordering rules to get wrong.
    """

    match: str
    torch_dtype: str = "float16"
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    prompt_format: str = "raw"  # "raw" | "chat"
    #: sha256 of the chat template this profile was written against.  ``None``
    #: means "this model has no chat template", which is asserted, not assumed.
    chat_template_sha: str | None = None
    chat_template_kwargs: dict = field(default_factory=dict)
    notes: str = ""


def template_sha(tokenizer) -> str | None:
    """sha256 of a tokenizer's chat template, or ``None`` if it has none.

    Hashing the template rather than recording a version string is what makes
    drift detectable: HuggingFace revises templates in place, under the same
    model id, and nothing in the repo id changes when they do.
    """
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        return None
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def assert_compatible(profile: ModelProfile, tokenizer) -> None:
    """Raise unless the loaded tokenizer is the one the profile describes.

    Two branches, because base models and instruct models fail in opposite
    directions:

    * A **chat** profile requires a template, and requires it to be the pinned
      one.  Silently rendering against a revised template would change every
      training prompt without changing any config, which is exactly the class of
      bug ``_text_projection`` was written to prevent one level up.
    * A **raw** profile requires that no template is in play.  Llama-3.1-8B is a
      true base model with no post-training; chat-wrapping it would produce an
      input shape it has never seen, and this assertion is what stops that
      happening by accident later.

    Checked at config time rather than at first use, so it fails on any machine
    before a GPU is held.
    """
    found = template_sha(tokenizer)
    if profile.prompt_format == "chat":
        if found is None:
            raise ValueError(
                f"profile {profile.match!r} declares prompt_format='chat' but the "
                f"tokenizer has no chat_template. If the model ships its template "
                f"as a standalone .jinja file, check that the prefetch job's "
                f"allow_patterns includes '*.jinja'."
            )
        if profile.chat_template_sha is None:
            raise ValueError(
                f"profile {profile.match!r} is a chat profile with no "
                f"chat_template_sha pin; the loaded template hashes to {found}. "
                f"Pin it deliberately rather than tracking upstream silently."
            )
        if found != profile.chat_template_sha:
            raise ValueError(
                f"chat template drift for {profile.match!r}: expected "
                f"{profile.chat_template_sha}, loaded {found}. The template was "
                f"revised upstream. Re-render the prompts, confirm the "
                f"prompt/completion cut still lands on a special token, and "
                f"update the pin deliberately -- adapters trained against the old "
                f"template are not comparable to ones trained against the new."
            )
        return

    if found is not None:
        raise ValueError(
            f"profile {profile.match!r} declares prompt_format='raw' but the "
            f"tokenizer carries a chat template ({found}). A base model prompted "
            f"through a chat template sees an input shape it was never trained "
            f"on. Either use a chat profile or load the base tokenizer."
        )


def resolve(model_id: str) -> ModelProfile:
    """The profile for a model id, by longest matching prefix.

    Falls back to a bare default rather than raising: an unknown model is not an
    error, it just gets the conservative settings (fp16, q/v attention, raw
    prompts) that every model in the repo used before profiles existed.
    """
    from src.models.profiles import PROFILES

    best: ModelProfile | None = None
    for profile in PROFILES:
        if model_id.startswith(profile.match):
            if best is None or len(profile.match) > len(best.match):
                best = profile
    return best if best is not None else ModelProfile(match="")
