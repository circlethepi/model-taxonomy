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

from scripts._utils import load_config, hf_token
from src.datasets.recipe import DatasetRecipe, DatasetEntry


def build_datasets(cfg: dict, validate: bool = False) -> dict[str, DatasetRecipe]:
    """Build DatasetRecipe objects and save them to {output_dir}/datasets/.

    Returns a mapping of recipe name → DatasetRecipe.
    """
    from src.datasets.mixed_dataset import MixedDataset

    output_dir = Path(cfg["output_dir"])
    datasets_dir = output_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    token = hf_token(cfg)
    recipes: dict[str, DatasetRecipe] = {}

    for ds_cfg in cfg.get("datasets", []):
        name = ds_cfg["name"]
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
        recipe = DatasetRecipe(name=name, datasets=entries)
        recipe_path = datasets_dir / f"{name}.recipe.json"
        recipe.save(recipe_path)
        recipes[name] = recipe

        print(f"  [{name}]  hash={recipe.recipe_hash()}  ->  {recipe_path}")
        for entry in entries:
            print(f"    {entry.dataset_id}  split={entry.split}  weight={entry.weight}"
                  f"  text_field={entry.text_field}")

        if validate:
            n_samples = ds_cfg.get("n_samples", 100)
            print(f"    Validating: loading {n_samples} samples...", end=" ", flush=True)
            mixed = MixedDataset(recipe, total_samples=n_samples, seed=ds_cfg.get("seed", 42),
                                  hf_token=token)
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
