#!/bin/bash -eu
#SBATCH --job-name=prometeo

CORES_PER_NODE=32
NUMA_PER_RANK=2
RAM_PER_NODE=150000
GPUS_PER_NODE=4
FB_PER_GPU=14000

# Use hyperthreading for the runtime
IO_THREADS="${IO_THREADS:-4}"
UTIL_THREADS="${UTIL_THREADS:-4}"
BGWORK_THREADS="${BGWORK_THREADS:-16}"

cd $SLURM_SUBMIT_DIR
source "$HTR_DIR"/jobscripts/jobscript_shared.sh

mpiexec -np "$NUM_RANKS" --map-by ppr:"$RANKS_PER_NODE":node --bind-to none \
    -x LD_LIBRARY_PATH -x HTR_DIR -x REALM_BACKTRACE -x LEGION_FREEZE_ON_ERROR \
     $COMMAND

# Resources:
# 256GB RAM per node
# 2 NUMA domains per node
# 64 cores per NUMA domain
# 4 nVidia Volta V100

