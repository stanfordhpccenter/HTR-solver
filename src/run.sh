#!/bin/bash -eu

###############################################################################
# Helper functions
###############################################################################

function quit {
    echo "$1" >&2
    exit 1
}

###############################################################################
# Derived options
###############################################################################

# We build the command line in a string before executing it, and it's very hard
# to get this to work if the executable name contains spaces, so we punt.
if [[ "$EXECUTABLE" != "${EXECUTABLE%[[:space:]]*}" ]]; then
    quit "Cannot handle spaces in executable name"
fi

# Command-line arguments are passed directly to the job script. We need to
# accept multiple arguments separated by whitespace, and pass them through the
# environment. It is very hard to properly handle spaces in arguments in this
# mode, so we punt.
for (( i = 1; i <= $#; i++ )); do
    if [[ "${!i}" != "${!i%[[:space:]]*}" ]]; then
        quit "Cannot handle spaces in command line arguments"
    fi
done
export ARGS=$@

export WALLTIME="$(printf "%02d:%02d:00" $((MINUTES/60)) $((MINUTES%60)))"

if [ "$(uname)" == "Darwin" ]; then
    export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:$LEGION_DIR/bindings/regent/"
    if [[ ! -z "${HDF_ROOT:-}" ]]; then
        export DYLD_LIBRARY_PATH="${DYLD_LIBRARY_PATH:-}:$HDF_ROOT/lib"
    fi
else
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$LEGION_DIR/bindings/regent/"
    if [[ ! -z "${HDF_ROOT:-}" ]]; then
        export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$HDF_ROOT/lib"
    fi
fi


# Make sure the number of requested ranks is divisible by the number of nodes.
export NUM_NODES=$(( NUM_RANKS / RANKS_PER_NODE ))
if (( NUM_RANKS % RANKS_PER_NODE > 0 )); then
   export NUM_NODES=$(( NUM_NODES + 1 ))
fi

export NUM_RANKS=$(( NUM_NODES * RANKS_PER_NODE ))

export LOCAL_RUN=0

export LEGION_FREEZE_ON_ERROR="$DEBUG"

###############################################################################
# Machine-specific handling
###############################################################################

function run_lassen {
    GROUP="${GROUP:-stanford}"
    export QUEUE="${QUEUE:-pbatch}"
    DEPS=
    if [[ ! -z "$AFTER" ]]; then
        DEPS="-w done($AFTER)"
    fi
    bsub -G "$GROUP" \
         -nnodes "$NUM_NODES" -W "$MINUTES" -q "$QUEUE" $DEPS \
         "$HTR_DIR"/src/lassen.lsf
}

function run_certainty {
    export QUEUE="${QUEUE:-gpu}"
    RESOURCES=
    if [[ "$QUEUE" == "gpu" ]]; then
        RESOURCES="gpu:4"
    fi
    EXCLUDED="$(paste -sd ',' "$HTR_DIR"/src/blacklist/certainty.txt)"
    DEPS=
    if [[ ! -z "$AFTER" ]]; then
        DEPS="-d afterok:$AFTER"
    fi
    sbatch --export=ALL \
        -N "$NUM_RANKS" -t "$WALLTIME" -p "$QUEUE" --gres="$RESOURCES" $DEPS \
        --exclude="$EXCLUDED" \
        "$HTR_DIR"/src/certainty.slurm
}

function run_sapling {
    # Allocate up to 2 nodes, from n0002 up to n0003
    if (( NUM_RANKS == 1 )); then
        NODES="n0002"
    elif (( NUM_RANKS == 2 )); then
        NODES="n0002,n0003"
    else
        quit "Too many nodes requested"
    fi
    # Synthesize final command
    CORES_PER_NODE=12
    RAM_PER_NODE=30000
    GPUS_PER_NODE=2
    FB_PER_GPU=5000
    source "$HTR_DIR"/src/jobscript_shared.sh
    # Emit final command
    mpiexec -H "$NODES" --bind-to none \
            -x LD_LIBRARY_PATH -x HTR_DIR -x REALM_BACKTRACE -x LEGION_FREEZE_ON_ERROR \
            $COMMAND
    # Resources:
    # 40230MB RAM per node
    # 2 NUMA domains per node
    # 6 cores per NUMA domain
    # 2-way SMT per core
    # 2 Tesla C2070 GPUs per node
    # 6GB FB per GPU
}

function run_quartz {
   export QUEUE="${QUEUE:-pbatch}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   sbatch --export=ALL \
         -N "$NUM_RANKS" -t "$WALLTIME" -p "$QUEUE" $DEPS \
         -J prometeo \
         "$HTR_DIR"/src/quartz.slurm
   # Resources:
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
}

function run_local {
    if (( NUM_NODES > 1 )); then
        quit "Too many nodes requested"
    fi
    # Overrides for local, non-GPU run
    LOCAL_RUN=1
    USE_CUDA=0
    RESERVED_CORES=2
    # Synthesize final command
    CORES_PER_NODE="$(grep ^cpu\\scores /proc/cpuinfo | uniq |  awk '{print $4}')"
    RAM_PER_NODE="$(free -m | head -2 | tail -1 | awk '{print $2}')"
    RAM_PER_NODE=$(( RAM_PER_NODE / 2 ))
    source "$HTR_DIR"/src/jobscript_shared.sh
    $COMMAND
}

###############################################################################
# Switch on machine
###############################################################################

if [[ "$(uname -n)" == *"lassen"* ]]; then
    run_lassen
elif [[ "$(uname -n)" == *"certainty"* ]]; then
    run_certainty
elif [[ "$(uname -n)" == *"sapling"* ]]; then
    run_sapling
elif [[ "$(uname -n)" == *"quartz"* ]]; then
    run_quartz
else
    echo 'Hostname not recognized; assuming local machine run w/o GPUs'
    run_local
fi
