#!/bin/bash -eu
#SBATCH -J prometeo

if [[ "$QUEUE" == "short" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=36
   NUMA_PER_RANK=2
   RAM_PER_NODE=120000
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 8 cores per NUMA domain
elif [[ "$QUEUE" == "batch" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=36
   NUMA_PER_RANK=2
   RAM_PER_NODE=120000
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 12 cores per NUMA domain
else
   echo "Unrecognized queue $QUEUE" >&2
   exit 1
fi

source "$HTR_DIR"/jobscripts/jobscript_shared.sh

#srun --mpi=pmix -n "$NUM_RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
#   --cpus-per-task="$CORES_PER_RANK" --export=ALL \
#   $COMMAND

srun -n "$NUM_RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
   --cpus-per-task="$CORES_PER_RANK" --export=ALL \
   --export=ALL \
   $COMMAND

