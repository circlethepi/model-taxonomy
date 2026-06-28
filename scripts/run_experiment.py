"""Master experiment orchestrator.

Reads an experiment YAML and runs any combination of the four pipeline steps:
  build      → Step 1: build and save dataset recipes
  finetune   → Step 2: LoRA fine-tune base models
  extract    → Step 3: extract model representations (activations / outputs)
  taxonomy   → Step 4: compute distance matrices and geometry embeddings

Usage:
    # Run all steps end-to-end
    python scripts/run_experiment.py experiments/example.yaml

    # Run specific steps
    python scripts/run_experiment.py experiments/example.yaml --steps build finetune

    # Skip fine-tuning (use existing adapters)
    python scripts/run_experiment.py experiments/example.yaml --steps build extract taxonomy

    # Force re-run fine-tuning even if adapters exist
    python scripts/run_experiment.py experiments/example.yaml --steps finetune --force

    # Filter which taxonomies to extract / analyse
    python scripts/run_experiment.py experiments/example.yaml --steps extract taxonomy \\
        --taxonomy functional behavioral
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


class _Tee:
    """Write to both a stream and a log file simultaneously."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import (
    load_config,
    expand_dataset_seeds,
    expand_dataset_n_samples,
    compute_recipe_capacity,
    hf_token,
)

ALL_STEPS = ["build", "finetune", "extract", "taxonomy"]


def apply_dataset_capacity_caps(cfg: dict) -> dict:
    """Cap each dataset's n_samples to the true recipe capacity and rename it.

    After ``expand_dataset_n_samples`` produces multiple blocks at different
    sizes, some may exceed how many samples the underlying data can actually
    deliver while maintaining class/entry proportions.  This function:

    1. Computes the capacity for each unique recipe (loading dataset sizes from
       HuggingFace cache; typically free after the first ``build`` step).
    2. For any block whose n_samples exceeds the capacity, reduces n_samples
       to the capacity and renames the block (``_n200000_`` → ``_n186667_``).
    3. Removes duplicate blocks that collapse to the same name after renaming,
       keeping only the first occurrence (lowest original n_samples).
    4. Emits a UserWarning for every change and every removed duplicate.

    Returns a new cfg dict (does not mutate the input).
    """
    import copy
    import re
    import warnings
    from scripts._utils import load_recipe as _load_recipe
    from pathlib import Path as _Path

    cfg = copy.deepcopy(cfg)
    output_dir = _Path(cfg["output_dir"])
    token = hf_token(cfg)

    # Build recipe objects grouped by recipe hash so we only compute capacity once.
    recipe_capacity_cache: dict[str, int] = {}

    updated: list[dict] = []
    seen_names: set[str] = set()

    for ds in cfg.get("datasets", []):
        recipe_path = output_dir / "datasets" / f"{ds['name']}.recipe.json"

        # If the recipe file doesn't exist yet (build step not run), skip capping.
        if not recipe_path.exists():
            if ds["name"] not in seen_names:
                seen_names.add(ds["name"])
                updated.append(ds)
            continue

        recipe = _load_recipe(recipe_path)
        rhash = recipe.recipe_hash()

        if rhash not in recipe_capacity_cache:
            recipe_capacity_cache[rhash] = compute_recipe_capacity(recipe, hf_token=token)
        capacity = recipe_capacity_cache[rhash]

        n_samples = ds.get("n_samples")
        if n_samples is not None and capacity > 0 and n_samples > capacity:
            old_name = ds["name"]
            # Replace the _n{number}_ or _n{number} suffix in the name.
            new_name = re.sub(r"_n\d+(?=_|$)", f"_n{capacity}", old_name)
            warnings.warn(
                f"Dataset cap: recipe capacity is {capacity}. "
                f"Reducing n_samples from {n_samples} to {capacity} "
                f"and renaming '{old_name}' → '{new_name}'.",
                stacklevel=2,
            )
            ds = dict(ds)
            ds["name"] = new_name
            ds["n_samples"] = capacity

        if ds["name"] in seen_names:
            warnings.warn(
                f"Removing duplicate dataset '{ds['name']}' "
                f"(collapsed to same capacity as an earlier entry).",
                stacklevel=2,
            )
            continue

        seen_names.add(ds["name"])
        updated.append(ds)

    cfg["datasets"] = updated
    return cfg


def _banner(title: str) -> None:
    width = 72
    print("\n" + "─" * width)
    print(f"  {title}")
    print("─" * width)


def run_experiment(
    cfg: dict,
    steps: list[str],
    force_finetune: bool = False,
    only_taxonomies: list[str] | None = None,
) -> None:
    cfg = expand_dataset_seeds(expand_dataset_n_samples(cfg))
    cfg = apply_dataset_capacity_caps(cfg)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy experiment config into output directory for reproducibility
    # (only if the source config is not already inside output_dir)
    config_copy = output_dir / "experiment.yaml"
    if not config_copy.exists():
        import yaml
        config_copy.write_text(yaml.dump(cfg, allow_unicode=True, sort_keys=False))

    t0_total = time.perf_counter()

    if "build" in steps:
        _banner("Step 1 / 4 — Build datasets")
        from scripts.build_datasets import main as build_main
        t0 = time.perf_counter()
        build_main(cfg)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    if "finetune" in steps:
        _banner("Step 2 / 4 — Fine-tune LoRA adapters")
        from scripts.finetune_lora import main as finetune_main
        t0 = time.perf_counter()
        finetune_main(cfg, force=force_finetune)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    if "extract" in steps:
        _banner("Step 3 / 4 — Extract representations")
        from scripts.extract_reprs import main as extract_main
        t0 = time.perf_counter()
        extract_main(cfg, only_taxonomies=only_taxonomies)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    if "taxonomy" in steps:
        _banner("Step 4 / 4 — Taxonomy analysis")
        from scripts.run_taxonomy import main as taxonomy_main
        t0 = time.perf_counter()
        taxonomy_main(cfg, only_taxonomies=only_taxonomies)
        print(f"  ({time.perf_counter() - t0:.1f}s)")

    elapsed = time.perf_counter() - t0_total
    print(f"\n{'═' * 72}")
    print(f"  Experiment '{cfg.get('name', '?')}' complete  ({elapsed:.1f}s total)")
    print(f"  Results: {output_dir.resolve()}")
    print(f"{'═' * 72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a model-taxonomy experiment from a YAML config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=ALL_STEPS,
        default=ALL_STEPS,
        metavar="STEP",
        help=f"Steps to run (default: all). Choices: {', '.join(ALL_STEPS)}.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-training even if adapter checkpoints already exist.",
    )
    parser.add_argument(
        "--taxonomy",
        nargs="+",
        metavar="NAME",
        help="Restrict extraction and taxonomy steps to these taxonomy names.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    exp_name = cfg.get("name", Path(args.config).stem)
    log_dir = Path(cfg["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{exp_name}_{timestamp}.log"

    with open(log_path, "w") as log_f:
        sys.stdout = _Tee(sys.__stdout__, log_f)
        sys.stderr = _Tee(sys.__stderr__, log_f)
        try:
            print(f"Logging output to {log_path}")
            run_experiment(
                cfg,
                steps=args.steps,
                force_finetune=args.force,
                only_taxonomies=args.taxonomy,
            )
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
