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

The boundary that actually holds, stated exactly: **a profile may be read at
runtime for facts about a pinned checkpoint or template, never for choices that
belong to an experiment.**  Three fields are read at runtime today and all three
are facts of the first kind -- ``pad_token`` (``finetune_lora.py``),
``chat_template_sha`` via ``assert_compatible`` (there and in
``scripts/_utils.py``), and ``prompt_end_token`` (``_chat_projection`` at render
time).  For a template pinned by ``chat_template_sha`` the correct cut point is
*determined*, so editing it is a bugfix or an error, never a legitimate change to
a past experiment -- which is the property the rule exists to protect.

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
    #: Trainable LoRA parameter count this checkpoint should produce at the
    #: repo's rank, or ``None`` to report the count without asserting it.  Not
    #: derived, because it is a *claim about what the run intends* -- the number
    #: falls out of the target list, and the point of stating it is to catch a
    #: target list that silently matched fewer modules than expected.
    #:
    #: A bare integer is well defined only because there is exactly one rank in
    #: the repo: ``LORA_RANK = 16`` in ``scripts/gen_simplex3.py`` is a module
    #: constant, not a ``Suite`` field.  If rank ever becomes per-suite this has
    #: to become rank-keyed or move to ``Suite``.
    expected_lora_params: int | None = None
    #: Token to use for padding when the checkpoint declares none, spelled as a
    #: literal so it is visible next to the rest of the checkpoint's description.
    #: ``None`` keeps the generic fallback (pad = eos).
    #:
    #: Why this is worth a field: with pad == eos, anything that masks on the pad
    #: id also masks a genuine end-of-turn, so a completion-only run can be taught
    #: never to emit the token that ends its own turn.  Llama-3.x ships
    #: ``<|finetune_right_pad_id|>`` for exactly this and then does not set it as
    #: ``pad_token``, so the fallback fires on precisely the models that supply a
    #: better answer.
    pad_token: str | None = None
    #: Modules that must **not** be matched by ``lora_target_modules``.  Also a
    #: choice rather than a property: Qwen3.5's ``in_proj_a``/``in_proj_b`` are
    #: excluded deliberately, and an exclusion nobody checks is an exclusion that
    #: quietly stops holding when a target list is edited.
    excluded_lora_modules: tuple[str, ...] = ()
    #: Accept generations that run the whole ``max_new_tokens`` budget instead of
    #: finishing inside it.  Default ``False``, so the smoke harness's stage 5
    #: keeps its full force for every checkpoint that does not say otherwise.
    #:
    #: A choice, like the two fields above, and the most consequential of the
    #: three: it declares that this model's behavioral level embeds *truncated*
    #: text rather than whole answers, which is a real difference from the suites
    #: it will be compared against.  Setting it is only defensible next to a
    #: ``notes`` entry saying why the truncation was accepted rather than fixed.
    allow_truncated_generation: bool = False
    #: Where the generation prompt ends: the token to cut immediately after, or
    #: ``None`` to keep the template's full render.  Required keyword-only, so
    #: the *unset* state cannot exist -- omitting it is a ``TypeError`` at
    #: import, and a profile author has to have thought about it.
    #:
    #: This replaces a derivation.  ``render_prompt`` used to cut after whichever
    #: added-vocab token sat furthest right in the rendered text, on the theory
    #: that the trailing whitespace after the final role marker was all it would
    #: ever remove.  OLMo-2 disproved that: it emits ``<|user|>``/``<|assistant|>``
    #: as ordinary text, so the only added-vocab token in its prompt is the BOS at
    #: index 0, the cut landed at character 13, and the rendered training prompt
    #: was the bare string ``<|endoftext|>`` with the question discarded.  No
    #: derivation could have got that right -- there is no atomic token after the
    #: question to find -- so the cut point is stated instead of guessed.
    #:
    #: ``None`` is the safe direction, not the ignorant one: it keeps the whole
    #: render, so the question is always present, and if the untrimmed seam is
    #: not a token boundary ``encode_pair``'s assertion raises.  Discarding
    #: content now requires someone to name a token deliberately.
    #:
    #: The named token must be atomic -- in the tokenizer's added vocabulary --
    #: or the cut lands on text BPE will merge across; ``assert_compatible``
    #: checks that at config time.  It is a fact about the pinned template rather
    #: than a per-experiment choice, which is why it lives beside
    #: ``chat_template_sha`` and is read at render time; see the runtime-lookup
    #: note in ``docs/guides/model_profiles.md``.
    prompt_end_token: str | None = field(kw_only=True)
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
        if profile.prompt_end_token is not None:
            added = getattr(tokenizer, "get_added_vocab", None)
            vocab = added() if callable(added) else {}
            if profile.prompt_end_token not in vocab:
                raise ValueError(
                    f"profile {profile.match!r} declares prompt_end_token "
                    f"{profile.prompt_end_token!r}, which is not in this "
                    f"tokenizer's added vocabulary. The prompt/completion cut "
                    f"must land on an atomic token, or BPE merges across the "
                    f"seam and the training prompt stops being a prefix of what "
                    f"generate() is handed."
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
    # prompt_end_token=None: raw, so a cut point is not applicable --
    # ``render_prompt`` returns before ever reading the field.
    return best if best is not None else ModelProfile(
        match="", prompt_end_token=None
    )


def apply_pad_token(tokenizer, profile: ModelProfile) -> str:
    """Give *tokenizer* a pad token, and say which rule supplied it.

    One definition, called from both training and inference, because the two
    disagreeing is the failure this prevents: a run trained with a distinct pad
    and extracted with pad == eos masks different positions in the two halves of
    the same experiment.

    Three outcomes, in order: the checkpoint already declares one and nothing
    happens; the profile names one and it is used; neither, and pad falls back to
    eos, which is what the whole simplex3 suite trained under and is preserved
    exactly.
    """
    if tokenizer.pad_token is not None:
        return f"declared by the checkpoint ({tokenizer.pad_token})"

    if profile.pad_token is not None:
        tid = tokenizer.convert_tokens_to_ids(profile.pad_token)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is None or tid == unk:
            raise ValueError(
                f"profile {profile.match!r} names pad_token "
                f"{profile.pad_token!r}, which this tokenizer does not have. "
                f"A pad token that silently resolves to unk would pad with a "
                f"real word."
            )
        if tid == tokenizer.eos_token_id:
            raise ValueError(
                f"profile {profile.match!r} names pad_token {profile.pad_token!r}, "
                f"which is the same id as eos ({tid}). Naming it deliberately and "
                f"getting eos anyway defeats the point of the field."
            )
        tokenizer.pad_token = profile.pad_token
        return f"from profile {profile.match!r} ({profile.pad_token} = {tid})"

    tokenizer.pad_token = tokenizer.eos_token
    return f"fallback pad = eos ({tokenizer.eos_token})"
