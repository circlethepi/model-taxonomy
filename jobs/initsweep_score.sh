#!/bin/bash
# Score the initsweep -- see figures/fig_structural_sweep/sweep_initsweep.py and
# docs/notes/init_seed_sweep.md.
#
# Resources, and why:
#
#   --partition=med   CPU only. Every representation this reads is already in
#                     the shared cache; the work is adapter loads, distance
#                     matrices, MDS fits and Procrustes alignments.
#
#   --mem=128G        Set by the pool load, not by the arithmetic. Analyses B
#                     and C assemble all 160 adapters at once, and
#                     build_taxonomy_artifacts resolves every model's
#                     representations *before* consulting the cache
#                     (src/analysis/comparison.py:235-244), so a warm
#                     07_collections does not save the memory. An interactive
#                     160-model structural pool degraded sharply from ~138
#                     models onward (30 s/model against 100/s before it), which
#                     is the shape of memory pressure, so this asks for well
#                     over the measured need.
#
#   --time=4:00:00    The 160-model structural pool measured 916 s interactively
#                     under that degradation; functional and the two behavioral
#                     rows are cheaper per pair but there are four perspectives.
#
#   --cpus-per-task=8 BLAS is pinned to one thread (MODEL_TAXONOMY_THREADS), so
#                     these are for the metric loops inside
#                     build_taxonomy_artifacts, matching the sibling jobs.
#
# Usage:
#   sbatch jobs/initsweep_score.sh                              # all four rows
#   sbatch jobs/initsweep_score.sh --perspective structural_all_o \
#       --seed 0 --seed 1 --seed 2 --outdir /path/to/scratch     # one diagnostic
#
# Arguments are passed straight through to the script.
#SBATCH --job-name=initsweep_score
#SBATCH --partition=med
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/initsweep/logs/initsweep_score-%j.out

set -euo pipefail

REPO=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/.claude/worktrees/initsweep-scoring
mkdir -p /weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/initsweep/logs

source /weka/home/jhu/mohata1/miniforge3/etc/profile.d/conda.sh
conda activate taxonomy-env

cd "$REPO"
echo "host=$(hostname)  start=$(date -Is)"

python figures/fig_structural_sweep/sweep_initsweep.py "$@"

echo "end=$(date -Is)"
