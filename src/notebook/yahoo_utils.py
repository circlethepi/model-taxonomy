from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

YAHOO_RECIPE_RE = re.compile(r"^yahoo_(.+)_n(\d+)_s(\d+)$")


def scan_yahoo_cache(cache_root: Path | str) -> dict[str, dict[str, list]]:
    """Scan the dataset_embeddings cache for Yahoo recipes.

    Returns a dict keyed by proportion string (e.g. ``"100t0_000t1"``) with
    sorted lists of covered n values and seeds::

        {
            "100t0_000t1": {"n_values": [1, 2, 5, 10], "seeds": [0, 1, 2]},
            ...
        }
    """
    emb_dir = Path(cache_root) / "dataset_embeddings"
    groups: dict[str, dict[str, set]] = defaultdict(
        lambda: {"n_values": set(), "seeds": set()}
    )

    for recipe_json in emb_dir.glob("*/recipe.json"):
        try:
            name = json.loads(recipe_json.read_text()).get("name", "")
        except Exception:
            continue
        m = YAHOO_RECIPE_RE.match(name)
        if not m:
            continue
        proportion, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
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
    emb_dir = Path(cache_root) / "dataset_embeddings"
    groups: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))

    for recipe_json in emb_dir.glob("*/recipe.json"):
        try:
            name = json.loads(recipe_json.read_text()).get("name", "")
        except Exception:
            continue
        m = YAHOO_RECIPE_RE.match(name)
        if not m:
            continue
        proportion, n, seed = m.group(1), int(m.group(2)), int(m.group(3))
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
