#!/bin/bash
#SBATCH --job-name=yahoo_behavioral_answerfield
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_topics_mean_cosine/slurm-%j.out
#SBATCH --error=results/yahoo_topics_mean_cosine/slurm-%j.err

set -euo pipefail

cd /weka/scratch/cpriebe1/MO/model-taxonomy

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env

python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_behavioral_answerfield.yaml --steps build extract
