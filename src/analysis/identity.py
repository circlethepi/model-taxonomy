"""Reconciling model identifiers across taxonomy levels.

The model-level taxonomies (``structural``, ``functional``, ``behavioral``) are
keyed by **adapter path** — the directory the fine-tuned LoRA lives in::

    results/yahoo_topics/adapters/meta-llama--Llama-3.2-3B/yahoo_topic0_only_r16

``DatasetEmbeddingTaxonomy`` is keyed by **recipe ID** — the ``name`` field of a
dataset block in the experiment YAML::

    yahoo_topic0_only

Those are the same experimental object seen from two sides, but as strings they
never match, so a naive set intersection between the two returns nothing.  This
module maps one onto the other so both can enter the same analysis.

The mapping is authoritative, not guessed: ``scripts/finetune_lora.py`` writes
an ``experiment_meta.json`` beside every adapter recording the exact
``dataset_name`` it was trained on, which *is* the recipe ID.  Only when that
file is missing (an adapter moved out of its experiment tree, say) do we fall
back to parsing the directory name, which ``scripts/_utils.adapter_dir`` builds
as ``{dataset_name}_r{lora_rank}_i{lora_init_seed:02d}``.

Typical use::

    from src.analysis import correlation_table, recipe_id_for

    labels, table = correlation_table(profile, key=recipe_id_for)

or, to rewrite the identifiers once and use everything downstream unchanged::

    from src.analysis import relabel, recipe_id_for

    structural = relabel(profile.get("structural").distance_matrix, recipe_id_for)

Nothing here mutates its input or touches stored results — a ``DistanceMatrix``
loaded from disk keeps the identifiers it was saved with.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping

from src.core.protocols import ModelID

__all__ = ["recipe_id_for", "relabel", "id_overlap"]

# scripts/_utils.adapter_dir: f"{dataset_name}_r{lora_rank}_i{lora_init_seed:02d}",
# with the _i suffix absent in adapters trained before it was introduced.
_ADAPTER_DIR_RE = re.compile(r"^(?P<name>.+?)_r\d+(?:_i\d+)?$")


def recipe_id_for(model_id: ModelID) -> str:
    """Map an adapter path to the recipe ID of the dataset it was trained on.

    Returns *model_id* unchanged when it is not an adapter path — so this is
    safe to apply uniformly across a mixed collection, including one that
    already contains recipe IDs, base-model HuggingFace IDs, or both.

    Resolution order:

    1. ``experiment_meta.json`` in the adapter directory, field ``dataset_name``.
       This is what the fine-tuning script recorded, so it is exact.
    2. The directory basename with the ``_r{rank}_i{seed}`` suffix removed.
    3. The input, unchanged.

    Note this is deliberately many-to-one: two adapters trained on the same
    dataset with different LoRA ranks or init seeds both map to that dataset's
    recipe ID.  When comparing against a dataset-level taxonomy that is exactly
    right — there is only one dataset — but it means a *within*-model-level
    collection that sweeps rank or seed must not be relabelled this way, since
    distinct models would collide onto one identifier.  :func:`relabel` raises
    if that happens rather than silently dropping rows.
    """
    text = str(model_id)

    meta = Path(text) / "experiment_meta.json"
    try:
        if meta.is_file():
            name = json.loads(meta.read_text()).get("dataset_name")
            if isinstance(name, str) and name:
                return name
    except (OSError, json.JSONDecodeError):
        pass  # unreadable or malformed — fall through to the name-based route

    # Only treat it as an adapter path if it looks like one: a trained adapter
    # always sits in a directory, so a bare HuggingFace ID like
    # "meta-llama/Llama-3.2-3B" must not be stripped down to "meta-llama".
    if "/" in text or "\\" in text:
        m = _ADAPTER_DIR_RE.match(Path(text).name)
        if m:
            return m.group("name")

    return text


def relabel(obj, key: Callable[[ModelID], str] | Mapping[ModelID, str]):
    """Return a copy of *obj* with its ``model_ids`` rewritten.

    Parameters
    ----------
    obj:
        A :class:`~src.core.distance.DistanceMatrix`, a
        :class:`~src.core.geometry.GeometryResult`, or a mapping of either
        (a mapping is rewritten value by value, keys untouched).
    key:
        A callable applied to each identifier, or a mapping to look them up in.
        Identifiers missing from a mapping are left as they are.

    The underlying arrays are shared, not copied — only the identifier list
    changes, so this is cheap enough to apply inline.

    Raises
    ------
    ValueError
        If the rewrite maps two distinct models onto the same identifier.  That
        would make the object's rows ambiguous, and silently keeping the first
        would quietly analyse the wrong thing.
    """
    if isinstance(obj, Mapping):
        return {k: relabel(v, key) for k, v in obj.items()}

    ids = getattr(obj, "model_ids", None)
    if ids is None:
        raise TypeError(
            f"relabel expects an object with model_ids (DistanceMatrix, "
            f"GeometryResult) or a mapping of them, got {type(obj).__name__}"
        )

    if isinstance(key, Mapping):
        new_ids = [key.get(m, m) for m in ids]
    else:
        new_ids = [key(m) or m for m in ids]

    if len(set(new_ids)) != len(new_ids):
        collisions = sorted({m for m in new_ids if new_ids.count(m) > 1})
        raise ValueError(
            f"relabel would map distinct models onto the same identifier: "
            f"{collisions}. This happens when a collection sweeps a parameter "
            f"the new identifier does not record — e.g. relabelling adapters to "
            f"recipe IDs when several ranks or init seeds share one dataset."
        )

    return replace(obj, model_ids=new_ids)


def id_overlap(*objs, key: Callable[[ModelID], str] | None = None) -> dict:
    """Report how well several objects' identifiers line up — a diagnostic.

    Useful when a comparison unexpectedly reports no common models: it shows
    each object's identifier count, the size of the intersection, and a sample
    of the identifiers that are unique to one side.

    With ``key=recipe_id_for`` it reports the overlap *after* relabelling, which
    is the quickest way to confirm a mapping does what you expect before
    committing to it.
    """
    apply = key or (lambda m: m)
    sets = [[apply(m) for m in getattr(o, "model_ids", o)] for o in objs]
    common = set(sets[0]).intersection(*(set(s) for s in sets[1:]))
    return {
        "sizes": [len(s) for s in sets],
        "n_common": len(common),
        "common": sorted(common),
        "only_in": [sorted(set(s) - common)[:5] for s in sets],
    }
