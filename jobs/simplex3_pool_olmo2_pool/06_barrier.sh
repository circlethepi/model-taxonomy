#!/bin/bash
#SBATCH --job-name=s3po2_barrier
#SBATCH --partition=med
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --time=0:05:00
#SBATCH --output=/weka/scratch/jhu/cpriebe1/MO/model-taxonomy/results/simplex3_pool_olmo2_pool/logs/s3po2_barrier-%j.out

# Pure synchronisation. Every training shard must finish before extraction
# starts, but a 251-long afterok list on each of 505 extraction jobs is both
# fragile (any shard that completes and is purged before submission finishes
# invalidates the whole list) and heavy on the scheduler. This job carries that
# dependency once; extraction then depends on this single id.
echo "all pool training shards completed at $(date -Is)"
