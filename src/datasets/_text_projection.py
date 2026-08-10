"""How a dataset row becomes one string, for both recipe types.

A recipe entry names the column its text comes from (``text_field``).  That was
enough while every use projected a row to a single column, and it stopped being
enough for ``docs/notes/TODO.md`` item 11: the yahoo adapters were trained on
bare ``best_answer`` prose — no question, no template — and then prompted at
extraction time with a ``question_title`` to continue.  The generations were
therefore produced from an input shape that never appeared in training, and the
behavioral level recovered the mixing order **exactly backwards**.

Training on ``question_title`` + ``best_answer`` pairs is the fix, and "a pair"
is not expressible as one column name.  So an entry may instead name
``text_fields`` — several columns, joined by ``text_separator``.

**Two rules make this safe to add to an existing cache.**

1. *The composition is part of the recipe, therefore part of ``recipe_hash``.*
   Two adapters trained on the same rows projected two different ways are not
   the same adapter, and the cache has to be able to tell them apart.

2. *An entry that does not use it must serialize exactly as before.*
   ``recipe_hash`` is a SHA-256 over ``to_dict()`` output, so emitting a new key
   unconditionally would change **every** existing hash at once — orphaning the
   523 draws in ``01_datasets``, all 25 adapters, and everything keyed on them.
   :func:`composition_dict` therefore returns an empty dict when no composition
   is set, and the caller splices it in with ``**``.  ``scripts/check_analysis.py``
   pins the six known recipe hashes so this cannot regress silently.
"""

from __future__ import annotations

#: What joins composed fields when the recipe does not say.  A bare newline, not
#: a template: behavioral extraction hands the model a question title and asks it
#: to continue, so ``"{question_title}\n{best_answer}"`` makes that exact shape
#: one the adapter saw in training.  A richer marker (``### Answer:``, a chat
#: template) would have to be mirrored on the extraction side or it reintroduces
#: the same mismatch one level up.
DEFAULT_SEPARATOR = "\n"


def composition_dict(text_fields: list[str] | None, text_separator: str) -> dict:
    """The composition keys for ``to_dict()``, or nothing at all.

    Empty when unset — see rule 2 in the module docstring.  This is the whole
    reason the change is additive rather than a migration.
    """
    if not text_fields:
        return {}
    return {"text_fields": list(text_fields), "text_separator": text_separator}


def read_composition(d: dict) -> tuple[list[str] | None, str]:
    """``(text_fields, text_separator)`` from a serialized entry."""
    fields = d.get("text_fields")
    return (list(fields) if fields else None), d.get("text_separator", DEFAULT_SEPARATOR)


def resolve_text(
    row: dict,
    text_field: str,
    text_fields: list[str] | None = None,
    separator: str = DEFAULT_SEPARATOR,
) -> str:
    """Project one dataset row to the string this entry means.

    Missing columns are skipped rather than rendered as ``"None"``: a row whose
    ``best_answer`` is absent should train on its question alone, not on the
    question followed by the word None.  A row missing *every* named column
    returns the empty string, which is what the single-field path already did.
    """
    if not text_fields:
        return str(row.get(text_field, ""))
    parts = [str(row[f]) for f in text_fields if row.get(f) is not None]
    return separator.join(parts)


def entry_text(entry, row: dict) -> str:
    """:func:`resolve_text` for a recipe entry, whichever recipe type it is.

    One implementation, so the training script, the query builder and the checks
    cannot disagree about what a row means.
    """
    return resolve_text(
        row,
        entry.text_field,
        getattr(entry, "text_fields", None),
        getattr(entry, "text_separator", DEFAULT_SEPARATOR),
    )


def row_text(recipe, row: dict) -> str:
    """The text one row contributes, under whichever entry claims it.

    A mixture may draw from datasets with different column names, so the entry
    is chosen by which of its named columns the row actually has, falling back to
    the first entry.  That is what the three copies of this loop in
    :mod:`src.datasets.mixed_dataset` did; it lives here now so that adding
    composition changed one implementation rather than three.
    """
    for entry in recipe.datasets:
        names = getattr(entry, "text_fields", None) or [entry.text_field]
        if any(n in row for n in names):
            return entry_text(entry, row)
    return entry_text(recipe.datasets[0], row)
