"""Step 2: fine-tune base models with LoRA adapters using PEFT + SFTTrainer.

Produces one adapter per (base_model, dataset) pair, saved to:
    {output_dir}/adapters/{base_model_slug}/{dataset_name}_r{lora_rank}_i{lora_init_seed:02d}/

The ``_s{seed}`` suffix in dataset names encodes the data-sampling seed;
the ``_i{seed}`` suffix in the adapter directory encodes the LoRA init seed.
A trailing ``_b{samples_seen}`` appears only when ``fine_tuning.total_train_samples``
set a sample budget, so that two runs differing only in training length do not
collide.  It records what the model actually saw — the budget rounded *up* to a
whole optimizer step — not what was requested.  Rounding up rather than to
nearest keeps the budget a floor: a run asking for 5000 samples is never trained
on fewer.

Usage:
    python scripts/finetune_lora.py experiments/example.yaml
    python scripts/finetune_lora.py experiments/example.yaml --force  # overwrite existing
    python scripts/finetune_lora.py experiments/example.yaml --dry-run  # resolve only
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
    get_adapter_root,
    hf_token,
    adapter_dir,
    load_recipe,
    make_mixed_dataset,
    make_sampled_dataset_cache,
    resolve_sample_budget,
    predicted_effective_batch,
    retag_adapter_dir,
    steps_for_budget,
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
    budget: int | None = None,
) -> Path | None:
    """Train one adapter.  Returns the directory it was written to — which is not
    necessarily *out_dir*, since a budgeted run is named for the samples it actually
    saw and that is only known once the Trainer exists.  None if it was skipped.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    import src.datasets._chat_projection as cp
    from src.models.profile import (
        apply_pad_token,
        assert_compatible,
        resolve as resolve_profile,
    )

    fmt = cp.PromptFormat.from_config(ft_cfg.get("_prompt_format"))
    format_id = fmt.format_id()

    if not force and out_dir.exists() and (out_dir / "adapter_config.json").exists():
        # An adapter directory carries its prompt format in its name (see
        # gen_simplex3.adapter_name), so reaching here with a different format
        # should be impossible.  Raise rather than skip anyway: if it ever does
        # happen, silently reusing an adapter fit on a different input shape is
        # the item-11 failure, and a skipped job looks exactly like a successful
        # one in sacct.
        meta_path = out_dir / "experiment_meta.json"
        if meta_path.exists():
            prior = json.loads(meta_path.read_text()).get("prompt_format_id")
            if prior != format_id:
                raise ValueError(
                    f"{out_dir} was trained with prompt_format_id={prior!r} but this "
                    f"run asks for {format_id!r}. These are different adapters that "
                    f"landed on the same path; fix the naming rather than overwriting."
                )
        print(f"    Already trained — skipping (use --force to retrain).")
        return None

    torch_dtype = getattr(torch, ft_cfg.get("torch_dtype", "float16"))

    print(f"    Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id, token=token, trust_remote_code=True
    )
    # pad == eos is the fallback, not the goal: anything masking on the pad id
    # also masks a genuine end-of-turn, so a completion-only run can be taught
    # never to end its own turn.  Qwen3.5 declares a distinct pad; Llama-3.x
    # ships <|finetune_right_pad_id|> without setting it, which is why the
    # profile can name one.  Llama-3.1-8B (base) names none and still trains
    # under pad == eos, exactly as the whole simplex3 suite did.
    profile = resolve_profile(base_model_id)
    print(f"[finetune] pad token: {apply_pad_token(tokenizer, profile)}", flush=True)

    # Fails here, before a GPU is held, if the checkpoint is not the one this
    # model's profile was written against -- a revised chat template upstream,
    # or a base model about to be chat-wrapped by mistake.
    assert_compatible(profile, tokenizer)

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
    lora_init_seed = ft_cfg.get("lora_init_seed", 0)
    torch.manual_seed(lora_init_seed)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    from src.datasets._text_projection import row_text

    recipe = load_recipe(recipe_path)
    n_samples = ft_cfg.get("n_samples", 1000)
    seed = ft_cfg.get("seed", 42)
    entry = recipe.datasets[0]
    text_fields = getattr(entry, "text_fields", None)
    text_field = entry.text_field
    described = f"{text_fields!r} joined by {entry.text_separator!r}" if text_fields else repr(text_field)

    print(f"    Building dataset: {n_samples} samples from '{dataset_name}' (field: {described})")
    mixed = make_mixed_dataset(recipe, total_samples=n_samples, seed=seed, hf_token=token,
                               sample_cache=sample_cache, name=dataset_name)
    rows = list(mixed.for_finetuning())

    max_len = ft_cfg.get("max_seq_length", 512)
    truncation_meta = None

    if fmt.format == "chat":
        # Tokenize here rather than handing trl two strings.  trl's
        # prompt/completion path uses the *joint* tokenization as input_ids and
        # only warns when the separately-tokenized prompt is not a prefix of it,
        # which would leave the training prompt ending in a token generate()
        # never produces.  encode_pair owns the split and asserts the two agree;
        # see src/datasets/_chat_projection for why the cut lands where it does.
        enc = [cp.encode_pair(tokenizer, row, fmt, max_len, recipe=recipe) for row in rows]
        n_trunc = sum(1 for e in enc if e["truncated"])
        n_empty = sum(1 for e in enc if e["n_completion_tokens"] == 0)
        if n_empty:
            # Truncation is keep_start, so what a long row loses is its answer.
            # Under completion-only loss a row with no answer left contributes
            # no supervised token at all; dropping it explicitly is honest,
            # where handing the trainer an all-masked example is not.
            print(f"    WARNING: {n_empty}/{len(enc)} rows have zero completion tokens "
                  f"at max_seq_length={max_len}; dropping them.")
        kept = [e for e in enc if e["n_completion_tokens"] > 0]
        truncation_meta = {
            "max_length": max_len,
            "mode": "keep_start",
            "rows_truncated": n_trunc,
            "rows_dropped_zero_completion": n_empty,
        }
        mean_sup = (sum(e["n_completion_tokens"] for e in kept) / len(kept)) if kept else 0
        hf_dataset = Dataset.from_list(
            [{"input_ids": e["input_ids"], "completion_mask": e["completion_mask"]}
             for e in kept]
        )
        # Must be explicit.  trl infers completion_only_loss from
        # ("prompt" in sample and "completion" in sample); a pre-tokenized
        # dataset has neither key, so the inference yields False and the run
        # would train full-sequence without saying so.
        sft_kwargs = dict(completion_only_loss=True)
        print(f"    Loss scope: COMPLETION ONLY (assistant turn) -- "
              f"prompt_format.format=chat, completion_only_loss=True; "
              f"{len(kept)} rows, mean {mean_sup:.0f} supervised tokens, "
              f"{n_trunc} truncated at {max_len}")
    else:
        if text_fields:
            # SFTTrainer takes ONE column name, so a composition has to become a
            # column.  Synthesized here rather than in the sampler so that what is
            # cached in 01_datasets stays the raw rows: the composition is a
            # projection of them, recorded in the recipe, not a different draw.
            #
            # This is the item 11 fix. Training on the bare answer column while
            # extraction prompts with a question is what put the behavioral level
            # out of distribution; the composed column is the shape both sides mean.
            text_field = "_composed_text"
            rows = [{**row, text_field: row_text(recipe, row)} for row in rows]

        hf_dataset = Dataset.from_list(rows)
        sft_kwargs = dict(dataset_text_field=text_field)
        print(f"    Loss scope: FULL SEQUENCE -- prompt_format absent (raw), "
              f"dataset_text_field={text_field}; {len(hf_dataset)} rows")

    n_epochs = ft_cfg.get("n_epochs", 3)
    sft_cfg = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=n_epochs,
        learning_rate=ft_cfg.get("learning_rate", 2e-4),
        per_device_train_batch_size=ft_cfg.get("per_device_train_batch_size", 4),
        gradient_accumulation_steps=ft_cfg.get("gradient_accumulation_steps", 4),
        max_length=max_len,
        bf16=(ft_cfg.get("torch_dtype") == "bfloat16"),
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        **sft_kwargs,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=hf_dataset,
        processing_class=tokenizer,
        args=sft_cfg,
    )

    # Sample budget: how many samples the model sees in total, independent of how
    # many distinct samples the dataset holds.  Expressed to the Trainer as a step
    # budget, because that is the only unit it accepts — which means the realized
    # count quantizes to the effective batch.  Trainer loops the dataloader across
    # as many epochs as max_steps needs, reshuffling each pass, so a budget far
    # larger than the dataset simply becomes many epochs.
    #
    # Read off trainer.args, not sft_cfg, and only after construction: Trainer
    # rewrites n_gpu (to 1) when device_map="auto" shards the model, and n_gpu is a
    # factor of train_batch_size.  This is the true effective batch, so this is the
    # number the adapter gets named for.  trainer.args *is* sft_cfg, so setting
    # max_steps here still reaches train().
    eff_batch = trainer.args.train_batch_size * trainer.args.gradient_accumulation_steps
    if budget is not None:
        max_steps = steps_for_budget(budget, eff_batch)
        trainer.args.max_steps = max_steps
        samples_seen = max_steps * eff_batch

        final_dir = retag_adapter_dir(out_dir, samples_seen)
        if final_dir != out_dir:
            print(
                f"    Effective batch is {eff_batch}, not the {predicted_effective_batch(ft_cfg)} "
                f"assumed when the directory was named; writing to {final_dir.name} instead."
            )
            # Trainer creates output_dir on construction; drop the empty one it made
            # for the name we are no longer using.
            if out_dir.is_dir() and not any(out_dir.iterdir()):
                out_dir.rmdir()
            out_dir = final_dir
            trainer.args.output_dir = str(out_dir)
            if not force and (out_dir / "adapter_config.json").exists():
                print(f"    Already trained at that budget — skipping (use --force to retrain).")
                return None
    else:
        max_steps = None
        samples_seen = len(hf_dataset) * n_epochs

    out_dir.mkdir(parents=True, exist_ok=True)

    if budget is None:
        print(f"    Training ({n_epochs} epoch(s), ~{samples_seen} samples seen)...")
    else:
        implied_epochs = samples_seen / len(hf_dataset) if len(hf_dataset) else 0.0
        note = (
            "" if samples_seen == budget
            else f" (budget {budget} rounded up to a step boundary)"
        )
        print(
            f"    Training ({max_steps} steps x effective batch {eff_batch} = "
            f"{samples_seen} samples seen, {implied_epochs:.2f} epoch(s) over "
            f"{len(hf_dataset)} rows){note}..."
        )
    trainer.train()
    trainer.save_model(str(out_dir))

    meta = {
        "base_model_id": base_model_id,
        "dataset_name": dataset_name,
        "recipe_hash": recipe.recipe_hash(),
        # What a row was projected to before it reached the trainer.  Inside
        # recipe_hash already, and repeated here because "what shape was this
        # adapter fit on?" is the first question asked of an adapter whose
        # generations look wrong, and it should not need a recipe lookup.
        "text_projection": (
            {"text_fields": text_fields, "text_separator": entry.text_separator}
            if text_fields else {"text_field": text_field}
        ),
        # How the row was wrapped before the model saw it.  Empty dict when raw,
        # so no existing adapter's metadata changes shape -- the same additive
        # rule as composition_dict in _text_projection.
        "prompt_format": fmt.to_dict(),
        "prompt_format_id": format_id,
        # Which chat template produced those strings.  HuggingFace revises
        # templates in place under an unchanged model id, so the model id alone
        # does not identify what this adapter was fit on.
        "chat_template_sha": cp.template_sha(tokenizer),
        "completion_only_loss": fmt.format == "chat",
        "truncation": truncation_meta,
        "lora_config": {
            "lora_rank": ft_cfg["lora_rank"],
            "lora_alpha": ft_cfg["lora_alpha"],
            "lora_init_seed": lora_init_seed,
            "target_modules": ft_cfg.get("target_modules"),
            "lora_dropout": ft_cfg.get("lora_dropout", 0.05),
        },
        "training": {
            "n_samples": n_samples,
            # The data-sampling seed, which picks which rows this adapter saw.  Recorded
            # because it is no longer recoverable from the recipe: the recipe hash is
            # content-addressed and its name no longer carries _s{seed}.  The block name
            # in dataset_name still does, but that is a convention, not a guarantee, and
            # CacheIndex has no other fallback for seed.
            "seed": ft_cfg.get("seed", 42),
            # Passes actually made over the dataset.  Under a budget this is *not*
            # the configured n_epochs — max_steps overrides it — so recording the
            # config value here would claim 3 for a run that made 10, and would
            # break the n_samples * n_epochs identity that older adapters rely on.
            "n_epochs": (
                n_epochs if budget is None
                else round(samples_seen / len(hf_dataset), 6) if len(hf_dataset) else None
            ),
            "learning_rate": ft_cfg.get("learning_rate", 2e-4),
            # How much training this adapter actually got, which n_samples and
            # n_epochs together no longer determine once a budget is in play.
            # samples_seen is recorded on both paths so it is a uniform axis to
            # select models on; under a budget it is the realized count, which may
            # differ from the request by up to one optimizer step.
            "total_train_samples": ft_cfg.get("total_train_samples"),
            "total_train_samples_resolved": budget,
            # The configured value, kept because n_epochs above now reports what
            # happened rather than what was asked for, and under a budget the two
            # differ.
            "n_epochs_configured": n_epochs,
            "max_steps": max_steps,
            "effective_batch_size": eff_batch,
            "samples_seen": samples_seen,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "experiment_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"    Saved adapter to {out_dir}")
    return out_dir


def finetune_all(cfg: dict, force: bool = False, dry_run: bool = False) -> list[Path]:
    """Fine-tune all configured (base_model, dataset) pairs.

    Returns a list of adapter directory paths that were produced.  With *dry_run*,
    resolves each pair's sample budget and output path and prints them without
    loading a model — the cheap way to check a budget before spending GPU hours on
    it.  Its step counts come from :func:`predicted_effective_batch`, so on a node
    where the model ends up sharded across devices the real run uses a different
    effective batch and lands on a different directory.
    """
    ft_cfg = cfg.get("fine_tuning", {})
    if not ft_cfg.get("enabled", True):
        print("  Fine-tuning disabled (fine_tuning.enabled=false).")
        return []

    output_dir = Path(cfg["output_dir"])
    adapter_root = get_adapter_root(cfg)
    token = hf_token(cfg)
    datasets_dir = output_dir / "datasets"
    sample_cache = make_sampled_dataset_cache(get_cache_dir(cfg))

    produced: list[Path] = []

    # Resolve fine_tuning.datasets against the expanded cfg datasets.
    # If the listed names are pre-expansion base names (e.g. "yahoo_100t0_000t1"),
    # match any expanded dataset whose name starts with "{base}_", and inherit
    # per-dataset n_samples/seed from the expanded block.  Exact-name matches and
    # configs without a sweep fall through unchanged.
    ft_base_names = set(ft_cfg.get("datasets", []))
    resolved_datasets: list[dict] = []
    for ds in cfg.get("datasets", []):
        name = ds["name"]
        if name in ft_base_names:
            resolved_datasets.append(ds)
        else:
            for base in ft_base_names:
                if name.startswith(base + "_"):
                    resolved_datasets.append(ds)
                    break
    # Fallback: no expanded matches → use the base names directly (backward compat)
    if not resolved_datasets:
        resolved_datasets = [{"name": n} for n in ft_cfg.get("datasets", [])]

    pairs = [
        (base, ds)
        for base in cfg.get("base_models", [])
        for ds in resolved_datasets
    ]

    for base_model_id, ds_block in pairs:
        dataset_name = ds_block["name"]
        # Per-dataset overrides for n_samples and seed; fall back to ft_cfg globals
        merged_ft_cfg = {
            **ft_cfg,
            "n_samples": ds_block.get("n_samples", ft_cfg.get("n_samples", 1000)),
            "seed": ds_block.get("seed", ft_cfg.get("seed", 42)),
            "total_train_samples": ds_block.get(
                "total_train_samples", ft_cfg.get("total_train_samples")
            ),
            # Top-level, not under fine_tuning:, because the same block feeds
            # extraction (scripts/_utils.make_queries).  One key describing both
            # sides is what makes training shape and query shape structurally
            # unable to drift -- the docs/notes/TODO.md item 11 lesson.  Passed
            # down under a private name so _finetune_one's signature is unchanged.
            "_prompt_format": cfg.get("prompt_format"),
        }
        recipe_path = datasets_dir / f"{dataset_name}.recipe.json"
        if not recipe_path.exists():
            raise FileNotFoundError(
                f"Recipe '{dataset_name}' not found at {recipe_path}. "
                "Run build_datasets.py first."
            )
        # Resolved here rather than inside _finetune_one because the budget is part
        # of the adapter's identity: it goes in the directory name.
        recipe = load_recipe(recipe_path)
        budget = resolve_sample_budget(
            merged_ft_cfg["total_train_samples"], recipe, hf_token=token
        )
        # The adapter is named for the samples it saw, which quantizes the budget to
        # the effective batch — predicted here so the already-trained check can run
        # before a model is loaded, and corrected inside _finetune_one on the rare
        # setup where the prediction is wrong.
        import src.datasets._chat_projection as _chat_projection

        eff = predicted_effective_batch(merged_ft_cfg)
        steps = None if budget is None else steps_for_budget(budget, eff)
        out_dir = adapter_dir(
            adapter_root, base_model_id, dataset_name,
            ft_cfg["lora_rank"], ft_cfg.get("lora_init_seed", 0),
            samples_seen=None if steps is None else steps * eff,
            # Must match scripts/gen_simplex3.adapter_name, which builds the
            # same leaf for the extraction configs' model list.
            prompt_format_id=_chat_projection.PromptFormat.from_config(
                cfg.get("prompt_format")
            ).format_id(),
        )

        if dry_run:
            n = merged_ft_cfg["n_samples"]
            n_epochs = ft_cfg.get("n_epochs", 3)
            if budget is None:
                plan = f"epoch mode, {n_epochs} epoch(s) → ~{n * n_epochs} samples"
            else:
                seen = steps * eff
                rounded = (
                    "" if seen == budget
                    else f" (budget {budget} rounded up to a step boundary)"
                )
                plan = (
                    f"budget {budget} → {steps} step(s) x {eff} = {seen} samples, "
                    f"{seen / n:.2f} epoch(s){rounded}"
                )
            print(f"  {base_model_id}  x  {dataset_name}  (n_samples={n})")
            print(f"      {plan}")
            print(f"      → {out_dir}")
            produced.append(out_dir)
            continue

        print(f"  {base_model_id}  x  {dataset_name}")
        written = _finetune_one(base_model_id, dataset_name, recipe_path, out_dir, merged_ft_cfg,
                               token, force, sample_cache=sample_cache, budget=budget)
        produced.append(written or out_dir)

    return produced


def main(cfg: dict, force: bool = False, dry_run: bool = False) -> None:
    print("=== Step 2: Fine-tune LoRA adapters ===")
    paths = finetune_all(cfg, force=force, dry_run=dry_run)
    if paths:
        verb = "would be trained" if dry_run else "trained"
        print(f"Done. {len(paths)} adapter(s) {verb}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune LoRA adapters from an experiment YAML.")
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain even if an adapter already exists at the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve sample budgets and output paths and print them; train nothing.",
    )
    args = parser.parse_args()
    main(
        expand_dataset_seeds(expand_dataset_n_samples(load_config(args.config))),
        force=args.force,
        dry_run=args.dry_run,
    )
