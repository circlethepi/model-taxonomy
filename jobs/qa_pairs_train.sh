#!/bin/bash
#SBATCH --job-name=yahoo_qa_pairs_train
#SBATCH --partition=l40s,h100,nvl,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_qa_pairs/slurm-%j.out
#SBATCH --error=results/yahoo_qa_pairs/slurm-%j.err

# Training only. Split from extraction so each half backfills on its own
# walltime -- a 1-hour request fits gaps that a combined 3-hour one does not,
# and the queue is saturated enough that gap size is the binding constraint.
#
# 5 adapters x 1000 samples x 3 epochs at effective batch 16 = ~188 optimizer
# steps each.

set -euo pipefail

cd /weka/scratch/cpriebe1/MO/model-taxonomy

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env

python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_qa_model_train.yaml --steps build finetune
