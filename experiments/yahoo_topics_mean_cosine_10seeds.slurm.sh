#!/bin/bash
#SBATCH --job-name=yahoo_topics_mean_cosine_10seeds
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_topics_mean_cosine_10seeds/slurm-%j.out
#SBATCH --error=results/yahoo_topics_mean_cosine_10seeds/slurm-%j.err
#SBATCH --mail-user=mohata1@jh.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Change to project root regardless of where sbatch is called from.
SCRIPT_DIR="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"
cd "$SCRIPT_DIR/.."

conda activate taxonomy-env
python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_topics_mean_cosine_10seeds.yaml
