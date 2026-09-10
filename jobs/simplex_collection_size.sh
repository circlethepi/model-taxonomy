#!/bin/bash
#SBATCH --job-name=scs_sweep
#SBATCH --partition=med
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex_collection_size/logs/scs_sweep-%j.out

# The collection-size suite: pool matrices, the size sweep, and the figure.
#
# Batch rather than interactive for one reason: this is hours of CPU and an
# interactive session's node can change under it.  Two earlier attempts died
# that way, each after finishing a matrix -- the work was not lost, because
# `build_taxonomy_artifacts` writes every matrix into `07_collections`, but the
# run had to be restarted each time.
#
# Sized from what those attempts measured, not from a guess:
#   * `structural_all_qkvo` is the memory wall.  1003 adapters x 16 layers x 4
#     projections of LoRA factors is ~17 GB resident, so 96G leaves headroom for
#     the pairwise loop's float64 working set.
#   * 8 CPUs for 7 concurrent perspectives.  `--jobs 7` runs one process per
#     perspective; MDS does not thread, so this is the only parallelism there is.
#   * 12 h because the sweep's cost is ~1400 MDS fits at n=500 for the whole
#     suite, and the matrices themselves took ~14 min for the cheapest all-layer
#     structural row.
#
# Restartable: everything already in `07_collections` is read back in
# milliseconds, so re-submitting after a timeout resumes rather than repeats.

set -euo pipefail

REPO=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/.claude/worktrees/simplex-collection-size
mkdir -p /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex_collection_size/logs

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  start=$(date -Is)"

python figures/simplex_collection_size/make_figures.py --jobs 7

echo "end=$(date -Is)"
