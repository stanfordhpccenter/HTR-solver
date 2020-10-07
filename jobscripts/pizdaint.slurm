#!/bin/bash -eu
#SBATCH --job-name=prometeo
#SBATCH --constraint=gpu

CORES_PER_NODE=12
RAM_PER_NODE=50000
GPUS_PER_NODE=1
FB_PER_GPU=15000

source "$HTR_DIR"/jobscripts/jobscript_shared.sh

srun -n "$NUM_RANKS" --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$CORES_PER_RANK" \
     --export=ALL \
     $COMMAND

# Resources:
# 12 cores per node
# 2-way SMT per core
# 64GB RAM per node
# 1 Tesla P100 GPU per node
# 16GB FB per GPU
