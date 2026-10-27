#!/bin/bash
# Score the qbig_v2 tree by subsampling it along the query axis -- see
# figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2.py.
#
# Resources, and why:
#
#   --partition=med   CPU only. Every representation this reads is already in
#                     the shared cache.
#
#   --mem=128G        The sampled matrix is 16 x 2000 x 64 x 768 float32 =
#                     6.3 GB, reduced in float64 beside it; the functional
#                     cross-check loads all states directly, 16 x 2000 x 17 x
#                     2048 float64 = 8.9 GB, and the pipeline loads its own
#                     copy. Twice qbig's footprint, against qbig's 96G.
#
#   --time=4:00:00    ~8,000 trials, each one MDS + Procrustes over 16 points;
#                     qbig's ~19,000-row run took 10:39 (job 1043594). The
#                     wall is charged only in queue priority.
#
#   --cpus-per-task=8 Matches the sibling scoring jobs.
#
# Usage:
#   sbatch jobs/qbig_v2_subsample.sh            # the full grid
#   sbatch jobs/qbig_v2_subsample.sh --quick    # a shape check
#SBATCH --job-name=qbig_v2_subsample
#SBATCH --partition=med
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_v2_olmo2_qbig_v2/logs/qbig_v2_subsample-%j.out

set -euo pipefail

# The worktree this job was submitted from, so a branch's code scores its own
# tree; falls back to the main checkout.
REPO=${SLURM_SUBMIT_DIR:-/weka/scratch/jhu/cpriebe1/MO/model-taxonomy}

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  repo=$REPO  start=$(date -Is)"
python -u figures/simplex3_qbig_v2_olmo2/subsample_qbig_v2.py "$@"
echo "end=$(date -Is)"
