#!/bin/bash -eu

#MSUB -N prometeo
#MSUB -l partition=quartz

USE_CUDA=0
CORES_PER_NODE=72
NUMA_PER_RANK=2
RAM_PER_NODE=100000

# 128GB RAM per node
# 2 NUMA domains per node
# 18 cores per NUMA domain
# 2-way SMT per core

source "$HTR_DIR"/jobscripts/jobscript_shared.sh

export OMPI_MCA_mtl="^psm,psm2"
srun -n "$NUM_RANKS" --tasks-per-node "$RANKS_PER_NODE" --cpu_bind none --mpibind=off \
   $COMMAND

