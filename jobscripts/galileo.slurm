#!/bin/bash -eu
#SBATCH --job-name=prometeo

if [[ "$QUEUE" == "gll_usr_prod" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=36
   NUMA_PER_RANK=2
   RAM_PER_NODE=70000
   # Resources:
   # 118GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
elif [[ "$QUEUE" == "gll_usr_gpuprod" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=36
   NUMA_PER_RANK=2
   RAM_PER_NODE=70000
   GPUS_PER_NODE=2
   FB_PER_GPU=10000
   # Resources:
   # 118GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
   # 1 nVidia K80 GPUs (seen as two K40 gpus)
else
    echo "Unrecognized queue $QUEUE" >&2
    exit 1
fi

cd $SLURM_SUBMIT_DIR
source "$HTR_DIR"/jobscripts/jobscript_shared.sh

srun -n "$NUM_RANKS" --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$CORES_PER_RANK" \
     --export=ALL \
     $COMMAND

# Resources:
# 118GB RAM per node
# 2 NUMA domains per node
# 18 cores per NUMA domain
# 1 nVidia K80 GPUs (seen as two K40 gpus)

