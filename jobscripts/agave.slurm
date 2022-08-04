#!/bin/bash -eu
#SBATCH --job-name=prometeo
#SBATCH --exclusive

if [[ "$QUEUE" == "parallel" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=27
   NUMA_PER_RANK=2
   RAM_PER_NODE=90000
   # Resources:
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 14 cores per NUMA domain
else
    echo "Unrecognized queue $QUEUE" >&2
    exit 1
fi

cd $SLURM_SUBMIT_DIR
source "$HTR_DIR"/jobscripts/jobscript_shared.sh

srun -n "$NUM_RANKS" \
     --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$CORES_PER_RANK" \
     --export=ALL --mpi=pmi2 \
     $COMMAND
