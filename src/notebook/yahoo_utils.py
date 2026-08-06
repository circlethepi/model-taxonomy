from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

YAHOO_RECIPE_RE = re.compile(r"^yahoo_(.+)_n(\d+)_s(\d+)$")


def parse_yahoo_recipe(
    recipe: str,
    field: str | None = None,
) -> dict[str, int | str] | int | str | None:
    """Parse a Yahoo recipe string into its components.

    Returns a dict with ``proportion``, ``size``, and ``seed`` keys,
    or a single value if *field* is one of those keys.
    Returns ``None`` if the string does not match the expected format.

    Examples::

        parse_yahoo_recipe("yahoo_100t0_000t1_n10_s3")
        # {"proportion": "100t0_000t1", "size": 10, "seed": 3}

        parse_yahoo_recipe("yahoo_100t0_000t1_n10_s3", field="seed")
        # 3
    """
    m = YAHOO_RECIPE_RE.match(recipe)
    if not m:
        return None
    parsed: dict[str, int | str] = {
        "proportion": m.group(1),
        "size": int(m.group(2)),
        "seed": int(m.group(3)),
    }
    if field is not None:
        return parsed[field]
    return parsed


def filter_yahoo_recipes(
    model_ids,
    keep: bool = False,
    props=None,
    sizes=None,
    seed=None,
) -> list[str]:
    """Filter an iterable of Yahoo recipe strings by proportion, size, and/or seed.

    A recipe *matches* when every provided (non-None) criterion agrees with the
    parsed recipe.  With all criteria ``None`` nothing matches, so the default
    call (``keep=False``) returns the original list unchanged.

    Args:
        model_ids: Iterable of Yahoo recipe strings.
        keep: If ``True``, return only matching recipes; if ``False`` (default),
            remove matching recipes and return the rest.
        props: Proportion string or iterable of proportion strings to match.
        sizes: Size value (``n``) or iterable of size values to match.
        seed: Seed value or iterable of seed values to match.
    """
    prop_set = ({props} if isinstance(props, str) else set(props)) if props is not None else None
    size_set = ({sizes} if isinstance(sizes, int) else set(sizes)) if sizes is not None else None
    seed_set = ({seed} if isinstance(seed, int) else set(seed)) if seed is not None else None

    has_filter = prop_set is not None or size_set is not None or seed_set is not None

    def matches(recipe_str: str) -> bool:
        if not has_filter:
            return False
        parsed = parse_yahoo_recipe(recipe_str)
        if parsed is None:
            return False
        if prop_set is not None and parsed["proportion"] not in prop_set:
            return False
        if size_set is not None and parsed["size"] not in size_set:
            return False
        if seed_set is not None and parsed["seed"] not in seed_set:
            return False
        return True

    return [r for r in model_ids if matches(r) == keep]


def _class_weights(recipe: dict) -> dict[str, float] | None:
    """A recipe's normalized class weights, or None if it has none.

    Read from the recipe body rather than parsed out of its *name*.  The name
    cannot be trusted: ``recipe_hash`` is content-addressed over
    ``{recipe_type, datasets}``, so one directory serves every draw of a mixture
    and its ``recipe.json`` keeps whichever name happened to write it first.  The
    class weights are part of what the hash is actually over.
    """
    datasets = recipe.get("datasets") or []
    if not datasets:
        return None
    weights = datasets[0].get("normalized_class_weights") or {}
    return {str(k): float(v) for k, v in weights.items()} or None


def _proportion_label(weights: dict[str, float], classes: list[str]) -> str:
    """``"075t0_025t1"`` over a fixed class universe.

    *classes* is passed in rather than taken from *weights* because a pure
    recipe filters to a single class and so records only that one — labelling it
    from its own keys alone would produce ``"100t0"`` where every other recipe
    produces ``"100t0_000t1"``, and the two would not sort or group together.
    """
    return "_".join(
        f"{round(weights.get(c, 0.0) * 100):03d}t{c}" for c in classes
    )


def _scan_yahoo_draws(cache_root: Path | str) -> list[tuple[str, int, int]]:
    """``[(proportion, n_samples, seed), ...]`` for every embedded Yahoo draw.

    Lists the ``n{n}_s{seed}`` directories that item 15 put in the path.  Before
    that the draw was inside ``embedder_hash`` and this had to be reconstructed
    from the recipe name, which reported **one** draw per proportion instead of
    the 103-135 actually stored — and dropped any proportion whose stored name
    predated the ``_n{n}_s{seed}`` convention entirely.
    """
    from src.cache._draw import parse_draw_name

    emb_dir = Path(cache_root) / "02_dataset_embeddings"
    if not emb_dir.is_dir():
        return []

    # Pass 1: which recipes carry class weights, and what is the class universe?
    found: list[tuple[Path, dict[str, float]]] = []
    classes: set[str] = set()
    for recipe_json in sorted(emb_dir.glob("*/recipe.json")):
        try:
            recipe = json.loads(recipe_json.read_text())
        except Exception:
            continue
        weights = _class_weights(recipe)
        if weights is None:
            continue
        found.append((recipe_json.parent, weights))
        classes.update(weights)

    ordered = sorted(classes, key=int)

    # Pass 2: list the draws each one has.
    out: list[tuple[str, int, int]] = []
    for recipe_dir, weights in found:
        proportion = _proportion_label(weights, ordered)
        for draw_dir in sorted(recipe_dir.iterdir()):
            parsed = parse_draw_name(draw_dir.name) if draw_dir.is_dir() else None
            if parsed:
                out.append((proportion, parsed[0], parsed[1]))
    return out


def scan_yahoo_cache(cache_root: Path | str) -> dict[str, dict[str, list]]:
    """Scan the dataset_embeddings cache for Yahoo recipes.

    Returns a dict keyed by proportion string (e.g. ``"100t0_000t1"``) with
    sorted lists of covered n values and seeds::

        {
            "100t0_000t1": {"n_values": [1, 2, 5, 10], "seeds": [0, 1, 2]},
            ...
        }
    """
    groups: dict[str, dict[str, set]] = defaultdict(
        lambda: {"n_values": set(), "seeds": set()}
    )
    for proportion, n, seed in _scan_yahoo_draws(cache_root):
        groups[proportion]["n_values"].add(n)
        groups[proportion]["seeds"].add(seed)

    return {
        prop: {
            "n_values": sorted(v["n_values"]),
            "seeds": sorted(v["seeds"]),
        }
        for prop, v in sorted(groups.items())
    }


def scan_yahoo_cache_detailed(
    cache_root: Path | str,
) -> dict[str, dict[int, list[int]]]:
    """Scan the dataset_embeddings cache for Yahoo recipes, tracking seeds per n value.

    Returns a nested dict ``{proportion: {n: [seeds]}}`` so you can see exactly
    which seeds are present for each individual n value::

        {
            "100t0_000t1": {1: [0,1,2], 2: [0,1], 5: [0]},
            ...
        }
    """
    groups: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for proportion, n, seed in _scan_yahoo_draws(cache_root):
        groups[proportion][n].add(seed)

    return {
        prop: {n: sorted(seeds) for n, seeds in sorted(n_map.items())}
        for prop, n_map in sorted(groups.items())
    }


def print_yahoo_coverage_detailed(cache_root: Path | str) -> None:
    """Print a per-n-value coverage table showing seeds for each (proportion, n) pair."""
    data = scan_yahoo_cache_detailed(cache_root)
    if not data:
        print("No Yahoo recipes found in cache.")
        return

    col1, col2 = 25, 9
    header = f"{'Class Proportions':{col1}} | {'n':>{col2}} | seeds"
    print(header)
    print("-" * len(header))
    for proportion, n_map in data.items():
        prop_display = proportion.replace("_", " ")
        for i, (n, seeds) in enumerate(n_map.items()):
            label = prop_display if i == 0 else ""
            print(f"{label:{col1}} | {n:>{col2}} | {seeds}")


def print_yahoo_coverage(cache_root: Path | str) -> None:
    """Print a coverage table showing which (proportion, n, seed) triples are cached."""
    data = scan_yahoo_cache(cache_root)
    if not data:
        print("No Yahoo recipes found in cache.")
        return

    col1, col2 = 25, 35
    header = f"{'Class Proportions':{col1}} | {'n values':{col2}} | seeds"
    print(header)
    print("-" * len(header))
    for proportion, info in data.items():
        prop_display = proportion.replace("_", " ")
        print(
            f"{prop_display:{col1}} | {str(info['n_values']):{col2}} | {info['seeds']}"
        )
