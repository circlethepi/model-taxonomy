#!/bin/bash
# Score the qbig tree by subsampling it along both probe axes, then draw its
# figures -- see figures/simplex3_qbig_olmo2/subsample_qbig.py and
# make_qbig_figures.py.
#
# Resources, and why:
#
#   --partition=med   CPU only. Every representation this reads is already in
#                     the shared cache; the work is cache reads, per-query
#                     inner products, MDS fits and Procrustes alignments.
#
#   --mem=96G         One embedder's sampled matrix is 16 x 1000 x 64 x 768
#                     float32 = 3.1 GB, and a full-probe replicate subset is
#                     gathered into float64 beside it (up to another ~3 GB);
#                     the functional all-states read is 16 x 1000 x 17 x 2048
#                     float64 = 4.5 GB. The pipeline cross-check loads each
#                     perspective a second time. ~20 GB peak is expected; this
#                     leaves wide headroom.
#
#   --time=12:00:00   ~5,500 cells, each one MDS + Procrustes over 16 points,
#                     plus ~550 replicate reductions at up to 1000 queries.
#                     Not measured end to end; matches qmix_score.sh.
#
#   --cpus-per-task=8 Matches the sibling scoring jobs; the batched GEMMs in
#                     cosine_parts use them.
#
# Usage:
#   sbatch jobs/qbig_subsample.sh                          # v2-moe, uniform draws
#   sbatch jobs/qbig_subsample.sh --embedders v15 v2moe    # both embedders
#
# Arguments are passed straight through to subsample_qbig.py.  The figures are
# redrawn from the CSV it writes.
#SBATCH --job-name=qbig_subsample
#SBATCH --partition=med
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_olmo2/logs/qbig_subsample-%j.out

set -euo pipefail

# The worktree this job was submitted from, so a branch's code scores its own
# tree; falls back to the main checkout.
REPO=${SLURM_SUBMIT_DIR:-/weka/scratch/jhu/cpriebe1/MO/model-taxonomy}
mkdir -p /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_olmo2/logs

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  repo=$REPO  start=$(date -Is)"

python -u figures/simplex3_qbig_olmo2/subsample_qbig.py "$@"
echo "scored=$(date -Is)"

python -u figures/simplex3_qbig_olmo2/make_qbig_figures.py

echo "end=$(date -Is)"
