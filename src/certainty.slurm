#!/bin/bash -eu
#SBATCH -J prometeo

GPU_OPTS=
if [[ "$QUEUE" == "all" ]]; then
    USE_CUDA=0
    CORES_PER_NODE=24
    RAM_PER_NODE=25000
elif [[ "$QUEUE" == "32core" ]]; then
    USE_CUDA=0
    CORES_PER_NODE=32
    RAM_PER_NODE=50000
    # 64GB RAM per node
    # 2 NUMA domains per node
    # 16 cores per NUMA domain
    # 2-way SMT per core
elif [[ "$QUEUE" == "24core" ]]; then
    USE_CUDA=0
    CORES_PER_NODE=24
    RAM_PER_NODE=25000
    # 36GB RAM per node
    # 2 NUMA domains per node
    # 12 cores per NUMA domain
    # 2-way SMT per core
elif [[ "$QUEUE" == "largemem" ]]; then
    USE_CUDA=0
    CORES_PER_NODE=24
    RAM_PER_NODE=120000
    # 144GB RAM per node
    # 2 NUMA domains per node
    # 12 cores per NUMA domain
    # 2-way SMT per core
elif [[ "$QUEUE" == "gpu" ]]; then
    USE_CUDA=1
    CORES_PER_NODE=16
    RAM_PER_NODE=25000
    GPUS_PER_NODE=4
    FB_PER_GPU=1024
    # 36GB RAM per node
    # 2 NUMA domains per node
    # 8 cores per NUMA domain
    # 2-way SMT per core
    # 4 Kepler M2050 GPUs per node
    # 3GB FB per GPU
else
    echo "Unrecognized queue $QUEUE" >&2
    exit 1
fi

source "$HTR_DIR"/src/jobscript_shared.sh

mpiexec -np "$NUM_RANKS" --map-by ppr:"$RANKS_PER_NODE":node --bind-to none \
    -x LD_LIBRARY_PATH -x HTR_DIR -x REALM_BACKTRACE -x LEGION_FREEZE_ON_ERROR \
    $COMMAND
