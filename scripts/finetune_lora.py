"""Step 2: fine-tune base models with LoRA adapters using PEFT + SFTTrainer.

Produces one adapter per (base_model, dataset) pair, saved to:
    {output_dir}/adapters/{base_model_slug}/{dataset_name}_r{lora_rank}/

Usage:
    python scripts/finetune_lora.py experiments/example.yaml
    python scripts/finetune_lora.py experiments/example.yaml --force  # overwrite existing
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import (
    load_config,
    expand_dataset_seeds,
    expand_dataset_n_samples,
    get_cache_dir,
    hf_token,
    adapter_dir,
    load_recipe,
    make_mixed_dataset,
    make_sampled_dataset_cache,
)


def _finetune_one(
    base_model_id: str,
    dataset_name: str,
    recipe_path: Path,
    out_dir: Path,
    ft_cfg: dict,
    token: str | None,
    force: bool = False,
    sample_cache=None,
) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    if not force and out_dir.exists() and (out_dir / "adapter_config.json").exists():
        print(f"    Already trained — skipping (use --force to retrain).")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    torch_dtype = getattr(torch, ft_cfg.get("torch_dtype", "float16"))

    print(f"    Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=token, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
        token=token,
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=ft_cfg["lora_rank"],
        lora_alpha=ft_cfg["lora_alpha"],
        target_modules=ft_cfg.get("target_modules", ["q_proj", "v_proj"]),
        lora_dropout=ft_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    recipe = load_recipe(recipe_path)
    n_samples = ft_cfg.get("n_samples", 1000)
    seed = ft_cfg.get("seed", 42)
    text_field = recipe.datasets[0].text_field

    print(f"    Building dataset: {n_samples} samples from '{dataset_name}' (field: {text_field!r})")
    mixed = make_mixed_dataset(recipe, total_samples=n_samples, seed=seed, hf_token=token,
                               sample_cache=sample_cache)
    hf_dataset = Dataset.from_list(list(mixed.for_finetuning()))

    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=ft_cfg.get("n_epochs", 3),
        learning_rate=ft_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=ft_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=ft_cfg.get("gradient_accumulation_steps", 4),
        max_length=ft_cfg.get("max_seq_length", 512),
        dataset_text_field=text_field,
        save_strategy="no",
        logging_steps=10,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        args=sft_cfg,
    )

    print(f"    Training ({ft_cfg.get('n_epochs', 3)} epoch(s))...")
    trainer.train()
    trainer.save_model(str(out_dir))

    meta = {
        "base_model_id": base_model_id,
        "dataset_name": dataset_name,
        "recipe_hash": recipe.recipe_hash(),
        "lora_config": {
            "lora_rank": ft_cfg["lora_rank"],
            "lora_alpha": ft_cfg["lora_alpha"],
            "target_modules": ft_cfg.get("target_modules"),
            "lora_dropout": ft_cfg.get("lora_dropout", 0.05),
        },
        "training": {
            "n_samples": n_samples,
            "n_epochs": ft_cfg.get("n_epochs", 3),
            "learning_rate": ft_cfg.get("learning_rate", 2e-4),
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    Saved adapter to {out_dir}")


def finetune_all(cfg: dict, force: bool = False) -> list[Path]:
    """Fine-tune all configured (base_model, dataset) pairs.

    Returns a list of adapter directory paths that were produced.
    """
    ft_cfg = cfg.get("fine_tuning", {})
    if not ft_cfg.get("enabled", True):
        print("  Fine-tuning disabled (fine_tuning.enabled=false).")
        return []

    output_dir = Path(cfg["output_dir"])
    token = hf_token(cfg)
    datasets_dir = output_dir / "datasets"
    sample_cache = make_sampled_dataset_cache(get_cache_dir(cfg))

    produced: list[Path] = []
    pairs = [
        (base, ds)
        for base in cfg.get("base_models", [])
        for ds in ft_cfg.get("datasets", [])
    ]

    for base_model_id, dataset_name in pairs:
        out_dir = adapter_dir(output_dir, base_model_id, dataset_name, ft_cfg["lora_rank"])
        recipe_path = datasets_dir / f"{dataset_name}.recipe.json"
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe '{dataset_name}' not found at {recipe_path}. "
                "Run build_datasets.py first."
            )
        print(f"  {base_model_id}  x  {dataset_name}")
        _finetune_one(base_model_id, dataset_name, recipe_path, out_dir, ft_cfg, token, force,
                      sample_cache=sample_cache)
        produced.append(out_dir)

    return produced


def main(cfg: dict, force: bool = False) -> None:
    print("=== Step 2: Fine-tune LoRA adapters ===")
    paths = finetune_all(cfg, force=force)
    if paths:
        print(f"Done. {len(paths)} adapter(s).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapters from an experiment YAML.")
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if an adapter already exists at the output directory.",
    )
    args = parser.parse_args()
    main(expand_dataset_seeds(expand_dataset_n_samples(load_config(args.config))), force=args.force)
