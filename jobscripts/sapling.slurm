#!/bin/bash -eu
#SBATCH -J prometeo

if [[ "$QUEUE" == "cpu" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=20
   NUMA_PER_RANK=2
   RAM_PER_NODE=220000
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 10 cores per NUMA domain
   # 2-way SMT per core
elif [[ "$QUEUE" == "gpu" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=20
   GPUS_PER_NODE=4
   NUMA_PER_RANK=2
   RAM_PER_NODE=220000
   FB_PER_GPU=14000
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 10 cores per NUMA domain
   # 2-way SMT per core
   # 4 Tesla P100 per node
else
   echo "Unrecognized queue $QUEUE" >&2
   exit 1
fi

# Use hyperthreading for the runtime
IO_THREADS="${IO_THREADS:-4}"
UTIL_THREADS="${UTIL_THREADS:-4}"
BGWORK_THREADS="${BGWORK_THREADS:-4}"

RESERVED_CORES=2

cd $SLURM_SUBMIT_DIR
source "$HTR_DIR"/jobscripts/jobscript_shared.sh

mpirun -n "$NUM_RANKS" -npernode "$RANKS_PER_NODE" --bind-to none \
    -x LD_LIBRARY_PATH -x HTR_DIR -x REALM_BACKTRACE -x LEGION_FREEZE_ON_ERROR \
     $COMMAND
