#!/bin/bash
#SBATCH --job-name=yahoo_qa_pairs_s[0]
#SBATCH --partition=l40s,h100,nvl,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_qa_pairs/slurm-%j.out
#SBATCH --error=results/yahoo_qa_pairs/slurm-%j.err

set -euo pipefail

cd /weka/scratch/cpriebe1/MO/model-taxonomy

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env

python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_qa_model_train.yaml --steps build finetune extract
