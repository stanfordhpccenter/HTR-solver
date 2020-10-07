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
    RAM_PER_NODE=200000
    # 256GB RAM per node
    # 2 NUMA domains per node
    # 12 cores per NUMA domain
    # 2-way SMT per core
else
    echo "Unrecognized queue $QUEUE" >&2
    exit 1
fi

source "$HTR_DIR"/jobscripts/jobscript_shared.sh

mpiexec -np "$NUM_RANKS" --map-by ppr:"$RANKS_PER_NODE":node --bind-to none \
    -x LD_LIBRARY_PATH -x HTR_DIR -x REALM_BACKTRACE -x LEGION_FREEZE_ON_ERROR \
    $COMMAND
