#!/bin/bash
# The label null for every cell figures/figure2_v2 plots -- see
# scripts/make_permutation_null.py and docs/notes/figure2_v2_permutation.md.
#
# Resources, and why:
#
#   --partition=med   CPU only. Nothing here touches a GPU: every
#                     representation it needs is already in the shared cache,
#                     and the work is distance matrices, MDS fits and
#                     permutation loops.
#
#   --mem=96G         Set by the pool load, not by the permutation work. The
#                     collection path calls sweep_group_size.pool_matrices over
#                     the 1003-model pool, and build_taxonomy_artifacts resolves
#                     every model's representations *before* consulting the
#                     cache (src/analysis/comparison.py:235-244), so a warm
#                     07_collections does not save the memory. This is the same
#                     figure jobs/simplex_collection_size.sh asks for, measured
#                     there at ~17 GB resident for the cheapest structural row.
#
#   --time=12:00:00   Also the pool load. With --dcor off the permutation
#                     arithmetic is negligible -- protest is a d x d SVD per
#                     draw, ~0.2 s at n=16 -- so the wall time is eight
#                     perspectives of pool assembly plus one MDS fit per group.
#                     Run with --probe first and set this from what it reports.
#
#   --cpus-per-task=8 BLAS is pinned to one thread (MODEL_TAXONOMY_THREADS), so
#                     these are for the metric loops inside
#                     build_taxonomy_artifacts, matching the sibling job.
#
# Usage:
#   sbatch jobs/figure2_v2_permutation_null.sh --probe        # measure, then stop
#   sbatch jobs/figure2_v2_permutation_null.sh --replicates 10
#
# Arguments are passed straight through to the script. The run is resumable:
# every cell already in 07A_permutation_tests is skipped, so a job that hits the
# time limit can simply be resubmitted.
#SBATCH --job-name=f2v2_perm
#SBATCH --partition=med
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/figure2_v2/logs/f2v2_perm-%j.out

set -euo pipefail

# The main checkout, not the worktree this was written in: that branch merged
# (a3116c8) and the worktree has sat at 4046310 since, which predates the
# lora_init_seed filter the init sweep made necessary. Run from there and the
# yahoo bars trip n_expected with 160 models before a single permutation.
REPO=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy
mkdir -p /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/figure2_v2/logs

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  start=$(date -Is)"

python scripts/make_permutation_null.py "$@"

echo "end=$(date -Is)"
