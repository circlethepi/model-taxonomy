"""How a dataset row becomes a prompt and a completion, for instruct models.

A **sibling** of :mod:`src.datasets._text_projection`, not an extension of it.
``row_text()`` is called by ``MixedDataset.to_queries``, which feeds the
dataset-embedding level; if chat rendering entered ``row_text`` the nomic
centroids would change, orphaning the 640 cached draws in
``02_dataset_embeddings`` and making the dataset level base-model-dependent,
which it is not.  ``_text_projection`` stays the pure, model-free row->string
projection.  This module is the model-aware layer above it, and it is the only
place in the repo that ever calls ``apply_chat_template``.

**The rule this module exists to enforce.**

    The prompt/completion boundary is a token boundary at inference, therefore it
    must be a token boundary at training -- and the cut belongs at an atomic
    token, so that tokenizing the two sides separately and tokenizing them
    together give the same ids.

``generate()`` is handed the prompt tokenized *alone* and appends to it.  So
training has to show the model a prompt prefix whose ids are byte-identical to
that.  Anything that tokenizes prompt and completion as one string breaks this
wherever BPE merges across the seam -- and it does merge.  Qwen3.5's generation
prompt ends ``<think>\\n`` and its completion begins ``\\n</think>``: separately
those are two ``\\n`` (id 198), jointly they are one ``\\n\\n`` (id 271).  trl
takes the joint tokenization (``sft_trainer.py``, the prompt-completion branch of
``tokenize_fn``) and only warns when the prefix check fails, which would leave
the training prompt ending in 271 where generation ends in 198.  That is the
``docs/notes/TODO.md`` item 11 mismatch, one level down at the tokenizer.

The fix is to own the tokenization *and* to move the cut.  Added tokens are never
merged into their neighbours, so cutting immediately after the last added token
of the generation prompt gives both properties at once::

    cut after "<think>\\n"   p=[...,248068,198]  c=[198,248069,...]   concat != joint
    cut after "<think>"      p=[...,248068]      c=[271,248069,...]   concat == joint

:func:`encode_pair` therefore *asserts* the equality rather than tolerating a
divergence rate.  The trailing whitespace moves from the prompt into the
completion; that is safe because :func:`render_prompt` is the single definition
of "the prompt" and extraction uses it too, so the two sides cannot disagree
about where the cut is.

None of this is Qwen-specific.  Llama-3 instruct's generation prompt ends
``<|end_header_id|>\\n\\n`` and takes the identical treatment.

**Additive by construction.**  ``PromptFormat()`` with no config block is
``format="raw"``, which delegates straight to ``row_text`` -- so every existing
experiment renders exactly the bytes it always did, and :meth:`PromptFormat.to_dict`
returns ``{}`` so no serialized config changes shape.  Same argument as rule 2 in
the ``_text_projection`` docstring, for the same reason.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from src.datasets._text_projection import DEFAULT_SEPARATOR, row_text


@dataclass(frozen=True)
class PromptFormat:
    """How a row is turned into model input, and under which template.

    ``user_fields`` and ``answer_fields`` are both plural and compose the same
    way: the assistant turn is built exactly as the user turn is, with missing
    columns skipped rather than rendered as the word ``None`` (the rule at
    ``_text_projection.resolve_text``).  A single-field answer is simply a
    one-element list.

    ``chat_template_kwargs`` are the *template's own* parameters -- Qwen's
    template reads ``enable_thinking``, for instance.  An empty dict means "take
    the template's defaults", which is why it is the default here.
    """

    format: str = "raw"  # "raw" | "chat"
    user_fields: tuple[str, ...] = ()
    answer_fields: tuple[str, ...] = ()
    separator: str = DEFAULT_SEPARATOR
    answer_separator: str = DEFAULT_SEPARATOR
    chat_template_kwargs: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, block: dict | None) -> "PromptFormat":
        """Read a ``prompt_format:`` block, or the raw default when absent."""
        if not block:
            return cls()
        fmt = block.get("format", "raw")
        if fmt not in ("raw", "chat"):
            raise ValueError(f"prompt_format.format must be 'raw' or 'chat', got {fmt!r}")
        if fmt == "chat" and not block.get("user_fields"):
            raise ValueError("prompt_format.format='chat' requires user_fields")
        return cls(
            format=fmt,
            user_fields=tuple(block.get("user_fields") or ()),
            answer_fields=tuple(block.get("answer_fields") or ()),
            separator=block.get("separator", DEFAULT_SEPARATOR),
            answer_separator=block.get("answer_separator", DEFAULT_SEPARATOR),
            chat_template_kwargs=dict(block.get("chat_template_kwargs") or {}),
        )

    def to_dict(self) -> dict:
        """Canonical serialization, or ``{}`` when raw.

        Empty when raw so that splicing this into an existing config with ``**``
        leaves that config byte-identical.  This is what keeps the change
        additive; see the module docstring.
        """
        if self.format == "raw":
            return {}
        return {
            "format": self.format,
            "user_fields": list(self.user_fields),
            "answer_fields": list(self.answer_fields),
            "separator": self.separator,
            "answer_separator": self.answer_separator,
            "chat_template_kwargs": self.chat_template_kwargs,
        }

    def format_id(self) -> str | None:
        """Short stable id for this format, or ``None`` when raw.

        ``None`` for raw is load-bearing: callers append this to draw and adapter
        names only when it is not None, so every existing path stays exactly as
        it is on disk.
        """
        d = self.to_dict()
        if not d:
            return None
        blob = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def template_sha(tokenizer) -> str | None:
    """sha256 of the tokenizer's chat template, or ``None``.

    Re-exported from :mod:`src.models.profile` so callers here need only one
    import; it is the same function and the same digest.
    """
    from src.models.profile import template_sha as _sha

    return _sha(tokenizer)


def _join_fields(row: dict, fields: tuple[str, ...], separator: str) -> str:
    """Compose named columns of a row, skipping the ones it does not have.

    Deliberately the same rule as ``_text_projection.resolve_text``: a row with
    no ``question_content`` should render as its title alone, not as the title
    followed by the word None.
    """
    parts = [str(row[f]) for f in fields if row.get(f) is not None]
    return separator.join(parts)


def _atomic_tokens(tokenizer) -> list[str]:
    """The tokens BPE will never merge across.

    ``all_special_tokens`` is *not* the right set.  Qwen3.5 puts ``<think>`` and
    ``<|im_start|>`` in the added vocabulary without listing them as special, yet
    they are single ids and are exactly as unmergeable as ``<|im_end|>`` is.
    What matters for the cut is atomicity, and that is what the added vocab
    describes.
    """
    added = getattr(tokenizer, "get_added_vocab", None)
    names = list(added().keys()) if callable(added) else []
    names.extend(getattr(tokenizer, "all_special_tokens", []) or [])
    return [n for n in dict.fromkeys(names) if n]


def _trim_to_last_atomic(tokenizer, text: str) -> str:
    """``text`` truncated to just past its last atomic token.

    Returns ``text`` unchanged if it contains none -- a template that ends in
    plain prose gets no trim, and :func:`encode_pair`'s assertion is then the
    thing that tells us whether the seam is safe.
    """
    end = -1
    for name in _atomic_tokens(tokenizer):
        idx = text.rfind(name)
        if idx >= 0:
            end = max(end, idx + len(name))
    return text if end < 0 else text[:end]


def render_prompt(tokenizer, row: dict, fmt: PromptFormat, recipe=None) -> str:
    """The exact string the model is prompted with, training and inference alike.

    This is the single definition of "the prompt".  ``make_queries`` calls it to
    build the extraction probes and :func:`render_pair` subtracts it from the
    full conversation to get the training completion, so the training prompt and
    the extraction prompt cannot drift apart -- which is the whole item-11
    lesson, stated as code rather than as a comment.
    """
    if fmt.format == "raw":
        if recipe is None:
            raise ValueError("raw prompt rendering needs the recipe that owns the row")
        return row_text(recipe, row)

    messages = [{"role": "user", "content": _join_fields(row, fmt.user_fields, fmt.separator)}]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **fmt.chat_template_kwargs,
    )
    return _trim_to_last_atomic(tokenizer, rendered)


def render_pair(tokenizer, row: dict, fmt: PromptFormat, recipe=None) -> tuple[str, str]:
    """``(prompt, completion)`` for one training row.

    The completion is taken by **subtraction** -- render the full conversation,
    render the prompt, and keep the difference -- rather than being assembled
    from template fragments by hand.  A future template revision therefore cannot
    silently desync the two sides: it either still starts with the prompt, or
    this raises.
    """
    if fmt.format == "raw":
        raise ValueError("render_pair is chat-only; the raw path trains on one column")

    prompt = render_prompt(tokenizer, row, fmt, recipe=recipe)
    messages = [
        {"role": "user", "content": _join_fields(row, fmt.user_fields, fmt.separator)},
        {"role": "assistant", "content": _join_fields(row, fmt.answer_fields, fmt.answer_separator)},
    ]
    full = tokenizer.apply_chat_template(
        messages, tokenize=False, **fmt.chat_template_kwargs
    )
    if not full.startswith(prompt):
        raise ValueError(
            "the rendered generation prompt is not a prefix of the rendered "
            "conversation; the chat template does not compose the way this "
            "module assumes and the prompt/completion split cannot be derived"
        )
    # Strip the template's trailing newline so the sequence ends exactly at the
    # end-of-turn token.  This also means the completion already ends with eos,
    # which is what stops trl's add_eos appending a second one.
    return prompt, full[len(prompt):].rstrip("\n")


def encode_pair(
    tokenizer,
    row: dict,
    fmt: PromptFormat,
    max_length: int,
    recipe=None,
) -> dict:
    """Tokenized ``input_ids`` plus the mask marking which tokens are supervised.

    The assertion is the point of this function.  Because :func:`render_prompt`
    cuts at an atomic token, tokenizing the two sides separately and
    concatenating must give exactly the joint tokenization -- so the prompt
    prefix is what ``generate()`` will see *and* the sequence is canonically
    tokenized.  If a template revision ever moves the seam onto mergeable text,
    this fails loudly instead of quietly training on ids the model will never be
    prompted with.

    ``add_special_tokens=False`` throughout: the template already emits every
    special token it wants, and adding more would duplicate them.

    Truncation is ``keep_start`` and explicit here rather than left to the
    trainer, because under completion-only loss a clipped row loses *supervised*
    tokens, and the caller needs the counts to report and to drop rows that end
    up with none.
    """
    prompt, completion = render_pair(tokenizer, row, fmt, recipe=recipe)
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids
    joint = tokenizer(prompt + completion, add_special_tokens=False).input_ids
    if p_ids + c_ids != joint:
        raise ValueError(
            "the prompt/completion cut is not a token boundary: tokenizing the "
            "sides separately does not reproduce the joint tokenization. The "
            "chat template's generation prompt no longer ends at an atomic "
            "token. See the module docstring."
        )

    ids = joint[:max_length]
    mask = ([0] * len(p_ids) + [1] * len(c_ids))[:max_length]
    return {
        "input_ids": ids,
        "completion_mask": mask,
        "n_prompt_tokens": len(p_ids),
        "n_completion_tokens": sum(mask),
        "truncated": len(joint) > max_length,
    }
