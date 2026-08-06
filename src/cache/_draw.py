"""One spelling of a draw, shared by every cache stage that names one.

A *draw* is a concrete sample from a recipe: ``(n_samples, seed)`` against a
``recipe_hash``.  Four stages record one, and until item 15 they did not agree
on how to write it down:

===========================  =========================  ====================
stage                        wrote                      as
===========================  =========================  ====================
``01_datasets``              ``n100000_s0.json``        a filename, unpadded
``02_dataset_embeddings``    *nothing*                  folded into a hash
``04_activations``           ``n64_s00``                a directory, padded
``05_generated``             ``n64_s00``                a directory, padded
===========================  =========================  ====================

Three schemes for one coordinate, so "which draws are here?" was a different
question at every stage — and at ``02`` it was not answerable by looking at all.
This module is the single answer: :func:`draw_name` builds the token and
:func:`parse_draw_name` reads it back, and every stage calls these rather than
formatting the string itself.

**The token is zero-padded** (``n{n}_s{seed:02d}``).  That was already what
``04``/``05`` wrote, so unifying on it left the larger caches untouched and moved
only ``01``.

**Parsing deliberately accepts both widths.**  ``01_datasets`` held unpadded
names before the migration, and a reader that could not open them would turn a
rename into data loss.  Writing is narrow, reading is wide.

This lives apart from :mod:`src.cache._draw_keyed` on purpose.  That module is
the base class for the *inference* caches, which are keyed by model as well as
by draw; ``01`` and ``02`` are keyed by recipe alone and have no business
importing an inference base class in order to spell a filename.
"""

from __future__ import annotations

import re

#: Matches a draw token, padded or not.  See the module docstring for why the
#: reader is deliberately more permissive than the writer.
DRAW_RE = re.compile(r"^n(\d+)_s(\d+)$")


def draw_name(n_samples: int, seed: int) -> str:
    """``n{n}_s{seed:02d}`` — the one way to write a draw down.

    Zero-padded on the seed only.  ``n_samples`` is not padded: it spans 1 to
    140,000 in the stored cache, so any fixed width would be either wrong or
    absurd, and no listing is sorted on it.
    """
    if seed is None:
        raise ValueError(
            "a draw needs a concrete seed; None would render as the literal "
            "'sNone', which names no draw and collides with nothing usefully"
        )
    return f"n{int(n_samples)}_s{int(seed):02d}"


def parse_draw_name(name: str) -> tuple[int, int] | None:
    """``(n_samples, seed)`` from a draw token, or None if it is not one.

    Accepts an unpadded seed so pre-migration ``01_datasets`` names still read.
    Returning None rather than raising is what lets callers walk a directory
    that holds non-draw entries — ``recipe.json``, ``names.json``, ``.lock`` —
    without special-casing each one.
    """
    m = DRAW_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))
