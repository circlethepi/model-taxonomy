#!/bin/bash
# Re-embed the qbig behavioral generations with nomic-embed-text-v2-moe.
#
# Reads the text already sitting in 05_generated and writes one new embeddings
# file per variant beside the existing v1.5 one.  No adapter is loaded and
# nothing is decoded: the 1,024,000 sampled continuations and 16,000 greedy ones
# were produced once and are cached, and `05_generated` splits text from vectors
# precisely so a second embedder costs a forward pass over text rather than a
# second generation run.
#
# Why this is not just a re-run of the behavioral config with the embedder
# swapped: `BehavioralTaxonomy.extract` tests `GeneratedTextCache.exists`, which
# requires the *embedding* for the requested embedder.  A draw whose text is
# present but whose v2-moe embedding is missing therefore falls through to
# `_extract_fresh` and re-decodes the lot -- ~5 GPU-hours to reproduce text that
# is already on disk, and, because sampling is seeded but batch-shape dependent,
# not even guaranteed to reproduce it exactly.  scripts/reembed_behavioral.py is
# the path that was missing.
#
# Submitted as an array: the 16 draw directories are split 8 ways by a stride,
# the tasks share nothing, and each write is atomic and idempotent -- so a task
# that dies is re-runnable on its own and the array can land in any order.
#
#   sbatch jobs/reembed_qbig_v2moe.sh
#
#SBATCH --job-name=s3bo2_reembed
#SBATCH --array=0-7
#SBATCH --partition=h200,h100,l40s,a100
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_olmo2_qbig/logs/s3bo2_reembed-%A_%a.out

set -euo pipefail

# Source conda's profile script directly, NOT ~/.bashrc -- a non-interactive
# shell (which is what SLURM gives a batch script) returns early from ~/.bashrc,
# so the `conda` shell function is never defined and `conda activate` dies with
# "Run 'conda init' before 'conda activate'".  Same incantation as the generated
# qbig jobs.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate taxonomy-env

REPO="${REEMBED_REPO:-/weka/scratch/jhu/cpriebe1/MO/model-taxonomy}"
cd "$REPO"

export TOKENIZERS_PARALLELISM=false
export HF_HOME=/weka/scratch/jhu/cpriebe1/MO/huggingface_cache

# --cache-root is passed explicitly rather than left to the default.  The script
# derives the default from its own location; run from a worktree that resolves
# inside the worktree, where no cache has ever been written, and the run would
# report an empty fleet rather than an unresolved path.  Naming it means a moved
# checkout fails loudly instead.
python scripts/reembed_behavioral.py \
    --cache-root /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/shared_cache \
    --base-model allenai/OLMo-2-0425-1B-Instruct \
    --draw n1000_s02_fea27ccee \
    --embedder-model nomic-ai/nomic-embed-text-v2-moe \
    --prompt-name search_document \
    --batch-size 256 \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 8
