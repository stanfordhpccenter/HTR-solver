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
   CI_FLAGS=
   if [[  "$CI_RUN" == 1 ]]; then
      CI_FLAGS="-K"
   fi
   bsub -G "$GROUP" $CI_FLAGS \
        -nnodes "$NUM_NODES" -W "$MINUTES" -q "$QUEUE" $DEPS \
        "$HTR_DIR"/jobscripts/lassen.lsf
}

function run_yellowstone {
   export QUEUE="${QUEUE:-gpu-maxwell}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
       DEPS="-d afterok:$AFTER"
   fi
   CI_FLAGS=
   if [[  "$CI_RUN" == 1 ]]; then
      CI_FLAGS="--wait"
   fi
   EXCLUDED="$(paste -sd ',' "$HTR_DIR"/jobscripts/blacklist/yellowstone.txt)"
   sbatch $CI_FLAGS --export=ALL \
       -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $DEPS \
       --exclude="$EXCLUDED" \
       "$HTR_DIR"/jobscripts/yellowstone.slurm
}

function run_armstrong {
   export QUEUE="${QUEUE:-compute}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
       DEPS="-d afterok:$AFTER"
   fi
   CI_FLAGS=
   if [[  "$CI_RUN" == 1 ]]; then
      CI_FLAGS="--wait"
   fi
   sbatch $CI_FLAGS --export=ALL \
       -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $DEPS \
       "$HTR_DIR"/jobscripts/armstrong.slurm
}

function run_sapling {
   if [[ "$USE_CUDA" == 1 ]]; then
      export QUEUE="${QUEUE:-gpu}"
   else
      export QUEUE="${QUEUE:-cpu}"
   fi
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   CI_FLAGS=
   if [[  "$CI_RUN" == 1 ]]; then
      CI_FLAGS="--wait"
   fi
   sbatch $CI_FLAGS --export=ALL --exclusive \
         -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $DEPS \
         "$HTR_DIR"/jobscripts/sapling.slurm
}

function run_quartz {
   export QUEUE="${QUEUE:-pbatch}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   CI_FLAGS=
   if [[  "$CI_RUN" == 1 ]]; then
      CI_FLAGS="--wait"
   fi
   sbatch $CI_FLAGS --export=ALL \
         -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $DEPS \
         -J prometeo \
         "$HTR_DIR"/jobscripts/quartz.slurm
   # Resources:
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
}

function run_solo {
   export QUEUE="${QUEUE:-pbatch}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   if [[  "$CI_RUN" == 1 ]]; then
      quit "CI is not suportted on this system yet"
   fi
   sbatch --export=ALL \
         -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $DEPS \
         -J prometeo \
         --account="$ACCOUNT" \
         "$HTR_DIR"/jobscripts/solo.slurm
   # Resources:
   # 128GB RAM per node
   # 2 NUMA domains per node
   # 18 cores per NUMA domain
}

function run_m100 {
   export QUEUE="${QUEUE:-m100_usr_prod}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
       DEPS="-w done($AFTER)"
   fi
   SPECIALQ=
   if [ $NUM_NODES -gt 16 ]; then
      SPECIALQ="--qos=m100_qos_bprod"
   fi
   if [[  "$CI_RUN" == 1 ]]; then
      quit "CI is not suportted on this system yet"
   fi
   CORES_PER_RANK=$(( 128/$RANKS_PER_NODE ))
   sbatch --export=ALL \
       -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" --gres=gpu:4 $DEPS $SPECIALQ \
       --ntasks-per-node="$RANKS_PER_NODE" --cpus-per-task="$CORES_PER_RANK" \
       --account="$ACCOUNT" \
        "$HTR_DIR"/jobscripts/m100.slurm
   # Resources:
   # 256GB RAM per node
   # 2 NUMA domains per node
   # 64 cores per NUMA domain
   # 4 nVidia Volta V100
}

function run_kraken {
   export QUEUE="${QUEUE:-gpua30}"
   DEPS=
   RESOURCES=
   if [[ "$QUEUE" == "gpua30" ]]; then
       RESOURCES="--gres=gpu:a30:4"
   fi
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   if [[  "$CI_RUN" == 1 ]]; then
      quit "CI is not suportted on this system yet"
   fi
   sbatch --export=ALL \
       -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" $RESOURCES $DEPS \
       "$HTR_DIR"/jobscripts/kraken.slurm
}

function run_agave {
   export QUEUE="${QUEUE:-parallel}"
   export PART="${PART:-normal}"
   DEPS=
   if [[ ! -z "$AFTER" ]]; then
      DEPS="-d afterok:$AFTER"
   fi
   sbatch --export=ALL \
       -N "$NUM_NODES" -t "$WALLTIME" -p "$QUEUE" -q "$PART" $DEPS \
       "$HTR_DIR"/jobscripts/agave.slurm
}

function run_local {
   if (( NUM_NODES > 1 )); then
      quit "Too many nodes requested"
   fi
   # Overrides for local, non-GPU run
   LOCAL_RUN=1
   USE_CUDA=0
   RESERVED_CORES=2
   # Determine available resources
   CORES_PER_NODE=
   THREADS_PER_NODE=
   RAM_PER_NODE=
   NUMA_PER_RANK=
   if [ "$(uname)" == "Darwin" ]; then
      CORES_PER_NODE="$(sysctl -a | grep machdep.cpu | awk -F ":" '/core_count/ { print $2; };')"
      THREADS_PER_NODE="$(sysctl -a | grep machdep.cpu | awk -F ":" '/thread_count/ { print $2; };')"
      RAM_PER_NODE="$(sysctl hw.memsize | awk '{ print $2/1048576; }')"
      NUMA_PER_RANK=1
   else
      CORES_PER_NODE="$(lscpu | awk -F ":" '/Core/ { c=$2; }; /Socket/ { print c*$2 }')"
      THREADS_PER_NODE="$(grep --count ^processor /proc/cpuinfo)"
      RAM_PER_NODE="$(free -m | head -2 | tail -1 | awk '{print $2}')"
      NUMA_PER_RANK="$(lscpu | awk -F ":" '/NUMA node\(s\)/ { gsub(/ /,""); print $2 }')"
   fi
   RAM_PER_NODE=$(( RAM_PER_NODE / 2 / RANKS_PER_NODE ))
   # Do not use more than one util or bgwork
   UTIL_THREADS=1
   BGWORK_THREADS=1
   # If the machine uses hyperthreading we do not need to reserve any core
   if (( CORES_PER_NODE != THREADS_PER_NODE )); then
      RESERVED_CORES=2
   fi
   # If we have multiple ranks per node, just set RESERVED_CORES to one
   if (( RANKS_PER_NODE > 1 )); then
      RESERVED_CORES=1
   fi
   # Generate command
   source "$HTR_DIR"/jobscripts/jobscript_shared.sh
   # Run
   if ((RANKS_PER_NODE > 1 )); then
      mpirun -n "$NUM_RANKS" --bind-to none $COMMAND
   else
      $COMMAND
   fi
}

###############################################################################
# Switch on machine
###############################################################################

if [[ "$(uname -n)" == *"lassen"* ]]; then
   run_lassen
elif [[ "$(uname -n)" == *"yellowstone"* ]]; then
   run_yellowstone
elif [[ "$(uname -n)" == *"armstrong"* ]]; then
   run_armstrong
elif [[ "$(uname -n)" == *"sapling"* ]]; then
   run_sapling
elif [[ "$(uname -n)" == *"quartz"* ]]; then
   run_quartz
elif [[ "$(uname -n)" == *"solo"* ]]; then
   run_solo
elif [[ "$(hostname -d)" == *"m100"* ]]; then
   run_m100
elif [[ "$(uname -n)" == *"kraken"* ]]; then
   run_kraken
elif [[ "$(uname -n)" == *"agave"* ]]; then
   run_agave
else
   echo 'Hostname not recognized; assuming local machine run w/o GPUs'
   run_local
fi
