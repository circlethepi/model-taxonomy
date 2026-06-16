"""Step 1: validate and persist DatasetRecipe objects from an experiment YAML.

Usage:
    python scripts/build_datasets.py experiments/example.yaml
    python scripts/build_datasets.py experiments/example.yaml --validate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from the repo root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import load_config, hf_token, make_mixed_dataset


def _build_recipe(ds_cfg: dict):
    """Build a DatasetRecipe or ClassAwareDatasetRecipe from a YAML dataset block."""
    name = ds_cfg["name"]
    recipe_type = ds_cfg.get("recipe_type", "simple")

    if recipe_type == "class_aware":
        from src.datasets.class_recipe import ClassAwareDatasetRecipe, ClassDatasetEntry
        entries = [
            ClassDatasetEntry(
                dataset_id=e["dataset_id"],
                split=e.get("split", "train"),
                weight=e.get("weight", 1.0),
                text_field=e.get("text_field", "text"),
                class_field=e.get("class_field", "label"),
                subset=e.get("subset"),
                class_filter=e.get("class_filter"),
                class_weights=e.get("class_weights"),
            )
            for e in ds_cfg["entries"]
        ]
        return ClassAwareDatasetRecipe(name=name, datasets=entries)
    else:
        from src.datasets.recipe import DatasetRecipe, DatasetEntry
        entries = [
            DatasetEntry(
                dataset_id=e["dataset_id"],
                split=e.get("split", "train"),
                weight=e.get("weight", 1.0),
                text_field=e.get("text_field", "text"),
                subset=e.get("subset"),
            )
            for e in ds_cfg["entries"]
        ]
        return DatasetRecipe(name=name, datasets=entries)


def build_datasets(cfg: dict, validate: bool = False) -> dict:
    """Build recipe objects and save them to {output_dir}/datasets/.

    Returns a mapping of recipe name → recipe object.
    """
    output_dir = Path(cfg["output_dir"])
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token(cfg)
    recipes = {}

    for ds_cfg in cfg.get("datasets", []):
        name = ds_cfg["name"]
        recipe = _build_recipe(ds_cfg)
        recipe_path = datasets_dir / f"{name}.recipe.json"
        recipe.save(recipe_path)
        recipes[name] = recipe

        rtype = ds_cfg.get("recipe_type", "simple")
        print(f"  [{name}]  type={rtype}  hash={recipe.recipe_hash()}  ->  {recipe_path}")
        for entry in recipe.datasets:
            line = (f"    {entry.dataset_id}  split={entry.split}"
                    f"  weight={entry.weight}  text_field={entry.text_field}")
            cf = getattr(entry, "class_filter", None)
            cw = getattr(entry, "class_weights", None)
            if cf is not None:
                line += f"  class_filter={cf}"
            if cw is not None:
                line += f"  class_weights={cw}"
            print(line)

        if validate:
            n_samples = ds_cfg.get("n_samples", 100)
            print(f"    Validating: loading {n_samples} samples...", end=" ", flush=True)
            mixed = make_mixed_dataset(
                recipe, total_samples=n_samples,
                seed=ds_cfg.get("seed", 42), hf_token=token,
            )
            queries = mixed.to_queries()
            print(f"OK  (got {len(queries)} queries, first: {queries[0][:60]!r})")

    return recipes


def main(cfg: dict, validate: bool = False) -> None:
    print("=== Step 1: Build datasets ===")
    recipes = build_datasets(cfg, validate=validate)
    print(f"Saved {len(recipes)} recipe(s) to {Path(cfg['output_dir']) / 'datasets'}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dataset recipes from an experiment YAML.")
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Actually load a sample from each dataset to verify it works.",
    )
    args = parser.parse_args()
    main(load_config(args.config), validate=args.validate)
