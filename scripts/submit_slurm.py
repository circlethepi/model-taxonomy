"""Generate a Slurm batch script for a model-taxonomy experiment.

Reads an experiment YAML, merges the optional ``slurm:`` section with CLI
overrides, and writes a ready-to-submit bash script.  The script can then be
submitted manually with ``sbatch <output.sh>``.

Usage examples::

    # Generate script with YAML defaults
    python scripts/submit_slurm.py experiments/yahoo_topics.yaml

    # Override resources from the command line
    python scripts/submit_slurm.py experiments/yahoo_topics.yaml \\
        --partition gpu_large --mem 160 --time 48:00:00

    # Generate for specific pipeline steps only
    python scripts/submit_slurm.py experiments/yahoo_topics.yaml \\
        --steps finetune extract --output jobs/finetune_only.sh

Slurm parameters are resolved in priority order (highest first):
  1. CLI flags (--partition, --mem, etc.)
  2. ``slurm:`` section in the experiment YAML
  3. Built-in defaults (partition=gpu, mem_gb=80, ...)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts._utils import load_config

# ── Defaults ───────────────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "partition": "gpu",
    "gpus_per_node": 1,
    "mem_gb": 80,
    "time": "24:00:00",
    "cpus_per_task": 8,
    "conda_env": "taxonomy-env",
    "modules": [],
    "account": None,
    "email": None,
    "mail_type": "END,FAIL",
}


# ── Script generation ──────────────────────────────────────────────────────────

def _build_slurm_params(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge defaults → YAML slurm section → CLI flags."""
    params = dict(_DEFAULTS)
    params.update(cfg.get("slurm", {}))

    # CLI overrides (only apply when the user explicitly passed the flag)
    cli_map = {
        "partition": args.partition,
        "gpus_per_node": args.gpus,
        "mem_gb": args.mem,
        "time": args.time,
        "cpus_per_task": args.cpus,
        "conda_env": args.conda_env,
        "account": args.account,
        "email": args.email,
        "mail_type": args.mail_type,
    }
    if args.modules:
        params["modules"] = args.modules
    for key, val in cli_map.items():
        if val is not None:
            params[key] = val

    return params


def _render_script(
    cfg: dict,
    slurm: dict,
    config_path: Path,
    run_args: list[str],
    script_path: Path,
) -> str:
    job_name = cfg.get("name", config_path.stem)
    output_dir = Path(cfg["output_dir"])

    # -- SBATCH header --
    lines = ["#!/bin/bash"]
    lines += [
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={slurm['partition']}",
        f"#SBATCH --gres=gpu:{slurm['gpus_per_node']}",
        f"#SBATCH --mem={slurm['mem_gb']}G",
        f"#SBATCH --time={slurm['time']}",
        f"#SBATCH --cpus-per-task={slurm['cpus_per_task']}",
        f"#SBATCH --output={output_dir}/slurm-%j.out",
        f"#SBATCH --error={output_dir}/slurm-%j.err",
    ]
    if slurm.get("account"):
        lines.append(f"#SBATCH --account={slurm['account']}")
    if slurm.get("email"):
        lines.append(f"#SBATCH --mail-user={slurm['email']}")
        lines.append(f"#SBATCH --mail-type={slurm['mail_type']}")

    lines.append("")
    lines.append("set -euo pipefail")
    lines.append("")

    # -- cd to project root (directory that contains this script) --
    lines.append('# Change to project root regardless of where sbatch is called from.')
    lines.append('SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"')
    lines.append('cd "$SCRIPT_DIR/.."')
    lines.append("")

    # -- optional module loads --
    modules = slurm.get("modules") or []
    if modules:
        for mod in modules:
            lines.append(f"module load {mod}")
        lines.append("")

    # -- optional conda activate --
    conda_env = slurm.get("conda_env")
    if conda_env:
        lines.append('source "$(conda info --base)/etc/profile.d/conda.sh"')
        lines.append(f"conda activate {conda_env}")
        lines.append("")

    # -- run experiment --
    run_cmd = ["python", "scripts/run_experiment.py", str(config_path)]
    run_cmd += run_args
    lines.append(" ".join(run_cmd))
    lines.append("")

    return "\n".join(lines)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Slurm batch script for a model-taxonomy experiment.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("config", help="Path to experiment YAML file.")
    parser.add_argument(
        "--output",
        metavar="PATH",
        help="Path for the generated .sh file. "
             "Default: <config_stem>.slurm.sh next to the YAML.",
    )

    # Pipeline pass-through args
    ppl = parser.add_argument_group("pipeline (passed to run_experiment.py)")
    ppl.add_argument(
        "--steps",
        nargs="+",
        choices=["build", "finetune", "extract", "taxonomy"],
        metavar="STEP",
        help="Steps to include (default: all four steps).",
    )
    ppl.add_argument(
        "--force",
        action="store_true",
        help="Add --force flag (re-run fine-tuning even if adapters exist).",
    )
    ppl.add_argument(
        "--taxonomy",
        nargs="+",
        metavar="NAME",
        help="Restrict to these taxonomy names.",
    )

    # Slurm resource overrides
    sl = parser.add_argument_group("slurm resource overrides")
    sl.add_argument("--partition", metavar="STR", help="Slurm partition.")
    sl.add_argument("--gpus", type=int, metavar="N", help="GPUs per node.")
    sl.add_argument("--mem", type=int, metavar="GB", help="Memory in GB.")
    sl.add_argument("--time", metavar="HH:MM:SS", help="Wall-clock time limit.")
    sl.add_argument("--cpus", type=int, metavar="N", help="CPUs per task.")
    sl.add_argument("--conda-env", metavar="NAME", help="Conda env to activate.")
    sl.add_argument("--modules", nargs="+", metavar="NAME", help="Modules to load.")
    sl.add_argument("--account", metavar="NAME", help="Slurm account/project.")
    sl.add_argument("--email", metavar="ADDR", help="Email address for job notifications.")
    sl.add_argument("--mail-type", metavar="EVENTS", default=None,
                    help="Slurm mail-type events (default: END,FAIL).")

    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_config(config_path)

    # Output path
    if args.output:
        script_path = Path(args.output)
    else:
        script_path = config_path.with_suffix(".slurm.sh")

    slurm = _build_slurm_params(cfg, args)

    # Build run_experiment.py extra args
    run_args: list[str] = []
    if args.steps:
        run_args += ["--steps"] + args.steps
    if args.force:
        run_args.append("--force")
    if args.taxonomy:
        run_args += ["--taxonomy"] + args.taxonomy

    script_text = _render_script(cfg, slurm, config_path, run_args, script_path)

    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(script_text)
    script_path.chmod(0o755)

    print(f"Generated: {script_path}")
    print(f"Submit with: sbatch {script_path}")


if __name__ == "__main__":
    main()
