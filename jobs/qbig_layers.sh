#!/bin/bash
# Score qbig's functional level per hidden state and per layer combination --
# see figures/simplex3_qbig_olmo2/layers_qbig.py.
#
# Resources, and why:
#
#   --partition=med   CPU only. The 17 per-layer activation files per adapter
#                     are already in the shared cache.
#
#   --mem=48G         One layer of all 16 adapters is 16 x 1000 x 2048 float64
#                     = 260 MB, decomposed and dropped before the next. The
#                     cross-check reads the all-states surrogate whole, 16 x
#                     1000 x 17 x 2048 float64 = 4.5 GB, plus the pipeline's own
#                     copy. ~12 GB peak is expected.
#
#   --time=04:00:00   26 surrogates x 91 query cells, each one MDS + Procrustes
#                     over 16 points, from precomputed parts. Not measured.
#
#   --cpus-per-task=8 Matches qbig_subsample.sh.
#
# Usage:
#   sbatch jobs/qbig_layers.sh
#SBATCH --job-name=qbig_layers
#SBATCH --partition=med
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_olmo2/logs/qbig_layers-%j.out

set -euo pipefail

REPO=${SLURM_SUBMIT_DIR:-/weka/scratch/jhu/cpriebe1/MO/model-taxonomy}
mkdir -p /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_qbig_olmo2/logs

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  repo=$REPO  start=$(date -Is)"

python -u figures/simplex3_qbig_olmo2/layers_qbig.py "$@"

echo "end=$(date -Is)"
