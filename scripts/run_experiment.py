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

from scripts._utils import load_config, expand_dataset_seeds, expand_dataset_n_samples, apply_dataset_size_caps, hf_token

ALL_STEPS = ["build", "finetune", "extract", "taxonomy"]


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
    cfg = apply_dataset_size_caps(cfg, hf_token=hf_token(cfg))
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
