#!/bin/bash
# Score the qmix_v2 tree by composing its pools -- see
# figures/simplex3_qmix_v2_olmo2/subsample_qmix_v2.py.
#
# Run only once all three pools are decoded: qbig_v2's yahoo pool and the two
# simplex3_qmix_v2_olmo2 pools (dolly 2000, oasst1cjk 859).
#
# Resources, and why:
#
#   --partition=med   CPU only. Every representation this reads is already in
#                     the shared cache.
#
#   --mem=128G        The largest single read is the yahoo pool's sampled
#                     matrix, 16 x 2000 x 64 x 768 float32 = 6.3 GB, reduced in
#                     float64 beside it, then freed; the functional checks load
#                     all states directly, 16 x 2000 x 17 x 2048 float64 =
#                     8.9 GB, once per pool. qbig_v2's footprint.
#
#   --time=6:00:00    ~129,000 rows, each one MDS + Procrustes over 16 points;
#                     a 1,035-row test over 100-row stand-in pools scored at
#                     ~28 rows/s, so ~80 min plus ~15 min of loading and checks.
#                     The wall is charged only in queue priority.
#
#   --cpus-per-task=8 Matches the sibling scoring jobs.
#
#   HF_* offline      The render check re-derives the yahoo draw, which loads
#                     yahoo_answers_topics and the OLMo tokenizer from the cache.
#
# Usage:
#   sbatch jobs/qmix_v2_subsample.sh            # the full grid
#   sbatch jobs/qmix_v2_subsample.sh --quick    # a shape check
#SBATCH --job-name=qmix_v2_subsample
#SBATCH --partition=med
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qmix_v2_olmo2/logs/qmix_v2_subsample-%j.out

set -euo pipefail

# The worktree this job was submitted from, so a branch's code scores its own
# tree; falls back to the main checkout.
REPO=${SLURM_SUBMIT_DIR:-/weka/scratch/jhu/cpriebe1/MO/model-taxonomy}

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env
export HF_HOME=/weka/scratch/jhu/cpriebe1/MO/huggingface_cache
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1

cd "$REPO"
echo "host=$(hostname)  repo=$REPO  start=$(date -Is)"
python -u figures/simplex3_qmix_v2_olmo2/subsample_qmix_v2.py "$@"
echo "end=$(date -Is)"
