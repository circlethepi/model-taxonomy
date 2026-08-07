#!/bin/bash
#SBATCH --job-name=yahoo_qa_pairs_extract
#SBATCH --partition=l40s,h100,nvl,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --output=results/yahoo_qa_pairs/slurm-%j.out
#SBATCH --error=results/yahoo_qa_pairs/slurm-%j.err

# Behavioral extraction over the five QA-pair adapters, R=8, prompted with
# question_title -- the same draw the re-extraction jobs use, so the two adapter
# sets land at the same coordinates and differ only in the adapter component.
#
# Depends on qa_pairs_train.sh: the adapters must exist, and its `build` step is
# what writes the yahoo_queries recipe into results/yahoo_qa_pairs/datasets/.
#
# Wall sized from the 2026-08-07 smoke (job 2029777): 8 queries x 2 replicates x
# 16 tokens took 22.5 s/model including load. This run is 64 x 8 x 128 over 5
# adapters -- 64x the decode steps per adapter -- putting it near an hour, so 2
# is padded without being the 4 that was guessed before any measurement existed.

set -euo pipefail

cd /weka/scratch/cpriebe1/MO/model-taxonomy

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env

python scripts/run_experiment.py /weka/scratch/cpriebe1/MO/model-taxonomy/experiments/yahoo_qa_model_train.yaml --steps extract
