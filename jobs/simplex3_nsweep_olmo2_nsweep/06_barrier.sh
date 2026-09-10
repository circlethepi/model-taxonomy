#!/bin/bash
#SBATCH --job-name=s3no2_barrier
#SBATCH --partition=med
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_nsweep_olmo2_nsweep/logs/s3no2_barrier-%j.out

# Pure synchronisation, the same device as the pool suite's 06_barrier.sh.
# Extraction must not start before the last training shard lands, but hanging a
# 30-id afterok list off each of 91 extraction jobs rots underneath the loop:
# a dependency naming a job SLURM has completed AND purged is rejected outright,
# and shards finish while the 91 are being submitted. This job carries the
# training dependency once, immediately; extraction depends on this single id.
echo "n=2000 and n=5000 training shards completed at $(date -Is)"
