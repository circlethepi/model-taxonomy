#!/bin/bash
#SBATCH --job-name=yahoo_topics_01_mix_prelim
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_topics/slurm-%j.out
#SBATCH --error=results/yahoo_topics/slurm-%j.err

set -euo pipefail

# Change to project root regardless of where sbatch is called from.
SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$SCRIPT_DIR/.."

python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_topics.yaml
