#!/bin/bash -eu
#SBATCH --job-name=prometeo
#SBATCH --exclusive

if [[ "$QUEUE" == "debug" ||
      "$QUEUE" == "prod"  ||
      "$QUEUE" == "prodshared" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=35
   NUMA_PER_RANK=2
   RAM_PER_NODE=80000
   # Resources:
   # 96GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
elif [[ "$QUEUE" == "gpu" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=35
   NUMA_PER_RANK=2
   RAM_PER_NODE=150000
   GPUS_PER_NODE=1
   FB_PER_GPU=12500
   # Resources:
   # 192GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
   # 1 nVidia Volta V100 GPU
elif [[ "$QUEUE" == "biggpu" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=35
   NUMA_PER_RANK=2
   RAM_PER_NODE=150000
   GPUS_PER_NODE=4
   FB_PER_GPU=29500
   # Resources:
   # 192GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
   # 4 nVidia Volta V100 GPU
elif [[ "$QUEUE" == "gpua30" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=35
   NUMA_PER_RANK=2
   RAM_PER_NODE=150000
   GPUS_PER_NODE=4
   FB_PER_GPU=22000
   # Resources:
   # 192GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
   # 4 nVidia Volta V100 GPU
else
   echo "Unrecognized queue $QUEUE" >&2
   exit 1
fi

cd $SLURM_SUBMIT_DIR
source "$HTR_DIR"/jobscripts/jobscript_shared.sh

srun -n "$NUM_RANKS" \
     --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$CORES_PER_RANK" \
     --export=ALL --mpi=pmi2 --cpu_bind=ldoms \
     $COMMAND
