#!/bin/bash -eu
#SBATCH -J prometeo

if [[ "$QUEUE" == "compute" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=16
   NUMA_PER_RANK=2
   RAM_PER_NODE=25000
   # 32GB RAM per node
   # 2 NUMA domains per node
   # 8 cores per NUMA domain
   # 2-way SMT per core
elif [[ "$QUEUE" == "largemem" ]]; then
   USE_CUDA=0
   CORES_PER_NODE=24
   NUMA_PER_RANK=2
   RAM_PER_NODE=220000
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 12 cores per NUMA domain
   # 2-way SMT per core
elif [[ "$QUEUE" == "gpu-k40" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=24
   NUMA_PER_RANK=2
   RAM_PER_NODE=220000
   GPUS_PER_NODE=1
   FB_PER_GPU=11000
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 12 cores per NUMA domain
   # 2-way SMT per core
   # 1 Kepler k40 GPU per node
   # 12GB FB per GPU
elif [[ "$QUEUE" == "gpu-maxwell" ]]; then
   USE_CUDA=1
   CORES_PER_NODE=24
   NUMA_PER_RANK=2
   RAM_PER_NODE=220000
   GPUS_PER_NODE=1
   FB_PER_GPU=10000
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 12 cores per NUMA domain
   # 2-way SMT per core
   # 1 Maxwell GeForce GTX Titan X GPU per node
   # 10GB FB per GPU
else
   echo "Unrecognized queue $QUEUE" >&2
   exit 1
fi

source "$HTR_DIR"/jobscripts/jobscript_shared.sh

srun --mpi=pmix -n "$NUM_RANKS" --ntasks-per-node="$RANKS_PER_NODE" \
   --cpus-per-task="$CORES_PER_RANK" --export=ALL \
   $COMMAND
